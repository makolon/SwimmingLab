from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING

import torch
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:  # circular-import safe
    from .lee_controller_cfg import (
        LeeControllerCfg,
        AttitudeControllerCfg,
        RateControllerCfg,
    )


def compute_mixer_and_limits(
    rotor_params: dict,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    angles = torch.as_tensor(rotor_params["rotor_angles"], device=device)
    arms = torch.as_tensor(rotor_params["arm_lengths"], device=device)
    f_const = torch.as_tensor(rotor_params["force_constants"], device=device)
    m_const = torch.as_tensor(rotor_params["moment_constants"], device=device)
    dirs = torch.as_tensor(rotor_params["directions"], device=device)
    max_rpm = torch.as_tensor(rotor_params["max_rotation_velocities"], device=device)

    A = torch.stack(
        [
            torch.sin(angles) * arms,
            -torch.cos(angles) * arms,
            -dirs * m_const / f_const,
            torch.ones_like(angles),
        ]
    )
    I_vec = torch.tensor(
        [
            rotor_params["inertia"]["xx"],
            rotor_params["inertia"]["yy"],
            rotor_params["inertia"]["zz"],
            1.0,
        ],
        device=device,
    )
    I = torch.diag_embed(I_vec)
    mixer = A.T @ torch.inverse(A @ A.T) @ I

    max_thrusts = max_rpm.square() * f_const
    mass = torch.tensor(rotor_params["mass"], device=device)
    I_inv = torch.inverse(I[:3, :3])
    return mixer, max_thrusts, mass, I_inv


class LeeController:
    """Full SE(3) Lee position and velocity controller (position + attitude + rate)."""

    def __init__(
        self,
        cfg: LeeControllerCfg,
        rotor_params: dict,
        num_envs: int,
        device: str,
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        self.mixer, self.max_thrusts, self.mass, self.I_inv = compute_mixer_and_limits(
            rotor_params, device
        )
        self.pos_gain = torch.as_tensor(cfg.position_gain, device=device)
        self.vel_gain = torch.as_tensor(cfg.velocity_gain, device=device)
        self.att_gain = torch.as_tensor(cfg.attitude_gain, device=device) @ self.I_inv[:3, :3]
        self.rate_gain = torch.as_tensor(cfg.angular_rate_gain, device=device) @ self.I_inv[:3, :3]
        self.g_vec = torch.tensor([0.0, 0.0, cfg.gravity], device=device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:  # high-level command dimension
        return 4

    """
    Operations.
    """

    def reset(self, env_ids: torch.Tensor = None):
        """Reset the internals.

        Args:
            env_ids: The environment indices to reset. If None, then all environments are reset.
        """
        pass

    def compute(
        self,
        root_state: torch.Tensor,
        target_cmd: torch.Tensor,
        body_rate: bool = False,
    ) -> torch.Tensor:
        bs, dev = root_state.shape[:-1], root_state.device
        target_pos = root_state[..., :3]
        target_vel = torch.zeros(*bs, 3, device=dev)
        target_acc = torch.zeros(*bs, 3, device=dev)
        if self.cfg.controller_type == "position":
            target_pos = target_cmd[:, :3]
        elif self.cfg.controller_type == "velocity":
            target_vel = target_cmd[:, :3]
        elif self.cfg.controller_type == "acceleration":
            target_acc = target_cmd[:, :3]
        target_yaw = target_cmd[:, 3:4]

        cmd = self._compute_impl(
            root_state.reshape(-1, 13),
            target_pos.reshape(-1, 3),
            target_vel.reshape(-1, 3),
            target_acc.reshape(-1, 3),
            target_yaw.reshape(-1, 1),
            body_rate,
        )
        return cmd.reshape(*bs, -1)

    def _compute_impl(self, state, tp, tv, ta, ty, body_rate):
        p, q, v, w = torch.split(state, [3, 4, 3, 3], -1)
        if not body_rate:
            w = math_utils.quat_rotate_inverse(q, w)

        pos_err = p - tp
        vel_err = v - tv
        acc_des = pos_err * self.pos_gain + vel_err * self.vel_gain - self.g_vec - ta

        R = math_utils.matrix_from_quat(q)
        b1_des = torch.cat([torch.cos(ty), torch.sin(ty), torch.zeros_like(ty)], -1)
        b3_des = -math_utils.normalize(acc_des)
        b2_des = math_utils.normalize(torch.cross(b3_des, b1_des, dim=-1))
        R_des = torch.stack(
            [torch.cross(b2_des, b3_des, dim=-1), b2_des, b3_des],
            -1,
        )

        ang_err_mat = 0.5 * (R_des.transpose(-2, -1) @ R - R.transpose(-2, -1) @ R_des)
        ang_err = torch.stack(
            [ang_err_mat[:, 2, 1], ang_err_mat[:, 0, 2], ang_err_mat[:, 1, 0]],
            -1,
        )
        ang_rate_err = w
        ang_acc = (
            -ang_err * self.att_gain
            - ang_rate_err * self.rate_gain
            + torch.cross(w, w, dim=-1)
        )

        thrust = -self.mass * (acc_des * R[:, :, 2]).sum(-1, keepdim=True)
        vec = torch.cat([ang_acc, thrust], -1)
        cmd = (self.mixer @ vec.T).T
        return (cmd / self.max_thrusts) * 2.0 - 1.0


class AttitudeController:
    """Roll-pitch-yaw controller with fixed thrust input."""

    def __init__(
        self,
        cfg: AttitudeControllerCfg,
        rotor_params: dict,
        num_envs: int,
        device: str,
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        self.mixer, self.max_thrusts, _, self.I_inv = compute_mixer_and_limits(rotor_params, device)
        self.att_gain = torch.as_tensor(cfg.gain_attitude, device=device) @ self.I_inv
        self.rate_gain = torch.as_tensor(cfg.gain_angular_rate, device=device) @ self.I_inv

    @property
    def action_dim(self) -> int:
        return 4

    def reset(self, env_ids: torch.Tensor | None = None):
        pass

    def compute(
        self,
        root_state: torch.Tensor,
        target_thrust: torch.Tensor,
        target_roll: torch.Tensor | None = None,
        target_pitch: torch.Tensor | None = None,
        target_yaw_rate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, dev = root_state.shape[:-1], root_state.device
        if target_roll is None:
            target_roll = torch.zeros(*bs, 1, device=dev)
        if target_pitch is None:
            target_pitch = torch.zeros(*bs, 1, device=dev)
        if target_yaw_rate is None:
            target_yaw_rate = torch.zeros(*bs, 1, device=dev)

        cmd = self._compute_impl(
            root_state.reshape(-1, 13),
            target_thrust.reshape(-1, 1),
            target_roll.reshape(-1, 1),
            target_pitch.reshape(-1, 1),
            target_yaw_rate.reshape(-1, 1),
        )
        return cmd.reshape(*bs, -1)

    def _compute_impl(self, state, T, roll, pitch, yaw_rate):
        _, q, _, ang = torch.split(state, [3, 4, 3, 3], -1)
        R = math_utils.matrix_from_quat(q)

        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0]).unsqueeze(-1)
        yaw_q = math_utils.quat_from_axis_angle(yaw, torch.tensor([0.0, 0.0, 1.0], device=q.device))
        roll_q = math_utils.quat_from_axis_angle(roll, torch.tensor([1.0, 0.0, 0.0], device=q.device))
        pitch_q = math_utils.quat_from_axis_angle(pitch, torch.tensor([0.0, 1.0, 0.0], device=q.device))

        R_des = (
            math_utils.matrix_from_quat(yaw_q)
            .bmm(math_utils.matrix_from_quat(roll_q))
            .bmm(math_utils.matrix_from_quat(pitch_q))
        )

        ang_err_mat = 0.5 * (R_des.transpose(-2, -1) @ R - R.transpose(-2, -1) @ R_des)
        ang_err = torch.stack(
            [ang_err_mat[:, 2, 1], ang_err_mat[:, 0, 2], torch.zeros_like(roll.squeeze(-1))],
            -1,
        )

        body_rate = math_utils.quat_rotate_inverse(q, ang)
        rate_des = torch.zeros_like(body_rate)
        rate_des[:, 2] = yaw_rate.squeeze(-1)
        rate_err = body_rate - rate_des

        ang_acc = (
            -ang_err * self.att_gain
            - rate_err * self.rate_gain
            + torch.cross(ang, ang, dim=-1)
        )
        vec = torch.cat([ang_acc, T], -1)
        cmd = (self.mixer @ vec.T).T
        return (cmd / self.max_thrusts) * 2.0 - 1.0


class RateController:
    """Body-rate controller with thrust input."""

    def __init__(
        self,
        cfg: RateControllerCfg,
        rotor_params: dict,
        num_envs: int,
        device: str,
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        self.mixer, self.max_thrusts, _, _ = compute_mixer_and_limits(rotor_params, device)
        self.rate_gain = torch.as_tensor(cfg.gain_angular_rate, device=device)

    @property
    def action_dim(self) -> int:
        return 4

    def reset(self, env_ids: torch.Tensor | None = None):
        pass

    def compute(
        self,
        root_state: torch.Tensor,
        target_rate: torch.Tensor,
        target_thrust: torch.Tensor,
    ) -> torch.Tensor:
        bs = root_state.shape[:-1]
        cmd = self._compute_impl(
            root_state.reshape(-1, 13),
            target_rate.reshape(-1, 3),
            target_thrust.reshape(-1, 1),
        )
        return cmd.reshape(*bs, -1)

    def _compute_impl(self, state, rate_des, thrust):
        _, q, _, ang = torch.split(state, [3, 4, 3, 3], -1)
        body_rate = math_utils.quat_rotate_inverse(q, ang)
        rate_err = body_rate - rate_des
        acc_des = -rate_err * self.rate_gain + torch.cross(ang, ang, dim=-1)

        vec = torch.cat([acc_des, thrust], -1)
        cmd = (self.mixer @ vec.T).T
        return (cmd / self.max_thrusts) * 2.0 - 1.0

