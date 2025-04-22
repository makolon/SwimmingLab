from __future__ import annotations

import torch
from typing import Optional, TYPE_CHECKING

from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from .lee_position_controller_cfg import LeePositionControllerCfg, AttitudeControllerCfg, RateControllerCfg


def compute_mixer_and_limits(rotor_params: dict, device: str) -> Tuple[torch.Tensor, torch.Tensor,torch.Tensor]:
    rotor_angles = torch.as_tensor(rotor_params["rotor_angles"], dtype=torch.float32, device=device)
    arm_lengths = torch.as_tensor(rotor_params["arm_lengths"], dtype=torch.float32, device=device)
    force_constants = torch.as_tensor(rotor_params["force_constants"], dtype=torch.float32, device=device)
    moment_constants = torch.as_tensor(rotor_params["moment_constants"], dtype=torch.float32, device=device)
    directions = torch.as_tensor(rotor_params["directions"], dtype=torch.float32, device=device)
    max_rot_vel = torch.as_tensor(rotor_params["max_rotation_velocities"], dtype=torch.float32, device=device)
    A = torch.stack([
        torch.sin(rotor_angles) * arm_lengths,
        -torch.cos(rotor_angles) * arm_lengths,
        -directions * moment_constants / force_constants,
        torch.ones_like(rotor_angles),
    ])
    inertia = rotor_params["inertia"]
    I = torch.diag_embed(torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"], 1.0], device=device))
    mixer = A.T @ torch.inverse(A @ A.T) @ I
    max_thrusts = max_rot_vel.square() * force_constants
    I_inv = torch.inverse(I[:3, :3])
    return mixer, max_thrusts, I_inv


class LeePositionController:

    def __init__(self, cfg: LeePositionControllerCfg, num_envs: int, device: str):
        # store inputs
        self.cfg = cfg
        self.num_envs = num_envs
        self._device = device

        self._mixer, self._max_thrusts, self._I_inv = compute_mixer_and_limits(cfg.rotor_params, device)
        self._mass = torch.tensor(cfg.mass, dtype=torch.float32, device=device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the controller's input command."""
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
        target_pos: Optional[torch.Tensor] = None,
        target_vel: Optional[torch.Tensor] = None,
        target_acc: Optional[torch.Tensor] = None,
        target_yaw: Optional[torch.Tensor] = None,
        body_rate: bool = False,
    ) -> torch.Tensor:
        bs = root_state.shape[:-1]
        dev = root_state.device
        if target_pos is None:
            target_pos = root_state[..., :3]
        if target_vel is None:
            target_vel = torch.zeros(*bs, 3, device=dev)
        if target_acc is None:
            target_acc = torch.zeros(*bs, 3, device=dev)
        if target_yaw is None:
            target_yaw = quaternion_to_euler(root_state[..., 3:7])[..., -1:]
        cmd = self._compute_impl(
            root_state.reshape(-1, 13),
            target_pos.reshape(-1, 3),
            target_vel.reshape(-1, 3),
            target_acc.reshape(-1, 3),
            target_yaw.reshape(-1, 1),
            body_rate,
        )
        return cmd.reshape(*bs, -1)

    def _compute_impl(self, root_state, tp, tv, ta, ty, body_rate):
        pos, rot, vel, w = torch.split(root_state, [3, 4, 3, 3], -1)
        if not body_rate:
            w = math_utils.quat_rotate_inverse(rot, w)
        pos_err = pos - tp
        vel_err = vel - tv
        acc_des = (
            pos_err * torch.tensor(self.cfg.position_gain, device=pos.device)
            + vel_err * torch.tensor(self.cfg.velocity_gain, device=pos.device)
            - torch.tensor([0, 0, self.cfg.gravity], device=pos.device)
            - ta
        )
        R = math_utils.matrix_from_quat(rot)
        b1_des = torch.cat([torch.cos(ty), torch.sin(ty), torch.zeros_like(ty)], -1)
        b3_des = -math_utils.normalize(acc_des)
        b2_des = math_utils.normalize(torch.cross(b3_des, b1_des, dim=-1))
        R_des = torch.stack([torch.cross(b2_des, b3_des, dim=-1), b2_des, b3_des], -1)
        ang_err_mat = 0.5 * (R_des.transpose(-2, -1).bmm(R) - R.transpose(-2, -1).bmm(R_des))
        ang_err = torch.stack([ang_err_mat[:, 2, 1], ang_err_mat[:, 0, 2], ang_err_mat[:, 1, 0]], -1)
        ang_rate_err = w
        ang_acc = (
            -ang_err * torch.tensor(self.cfg.attitude_gain, device=pos.device)
            -ang_rate_err * torch.tensor(self.cfg.angular_rate_gain, device=pos.device)
            + torch.cross(w, w, dim=-1)
        )
        thrust = -self._mass * (acc_des * R[:, :, 2]).sum(-1, keepdim=True)
        vec = torch.cat([ang_acc, thrust], -1)
        cmd = (self._mixer @ vec.T).T
        return (cmd / self._max_thrusts) * 2 - 1


class AttitudeController:

    def __init__(self, cfg: AttitudeControllerCfg, num_envs: int, device: str):
        # store inputs
        self.cfg = cfg
        self.num_envs = num_envs
        self._device = device

        self._mixer, self._max_thrusts, _ = compute_mixer_and_limits(cfg.rotor_params, device)
        self._mass = torch.tensor(cfg.mass, dtype=torch.float32, device=device)
        self._gain_att = torch.tensor(cfg.gain_attitude, dtype=torch.float32, device=device)
        self._gain_rate = torch.tensor(cfg.gain_angular_rate, dtype=torch.float32, device=device)

    """
    Properties.
    """
    
    @property
    def action_dim(self) -> int:
        """Dimension of the controller's input command."""
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
        target_thrust: torch.Tensor,
        target_roll: torch.Tensor | None = None,
        target_pitch: torch.Tensor | None = None,
        target_yaw_rate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs = root_state.shape[:-1]
        device = root_state.device
        if target_roll is None:
            target_roll = torch.zeros(*bs, 1, device=device)
        if target_pitch is None:
            target_pitch = torch.zeros(*bs, 1, device=device)
        if target_yaw_rate is None:
            target_yaw_rate = torch.zeros(*bs, 1, device=device)

        cmd = self._compute_impl(
            root_state.reshape(-1, 13),
            target_thrust.reshape(-1, 1),
            target_roll.reshape(-1, 1),
            target_pitch.reshape(-1, 1),
            target_yaw_rate.reshape(-1, 1),
        )
        return cmd.reshape(*bs, -1)

    def _compute_impl(self, state, T, roll, pitch, yaw_rate):
        pos, quat, vel, ang = torch.split(state, [3, 4, 3, 3], -1)
        R = math_utils.matrix_from_quat(quat)

        # desired rotation
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0]).unsqueeze(-1)
        yaw_quat = math_utils.quat_from_axis_angle(yaw, torch.tensor([0.0, 0.0, 1.0], device=quat.device))
        roll_quat = math_utils.quat_from_axis_angle(roll, torch.tensor([1.0, 0.0, 0.0], device=quat.device))
        pitch_quat = math_utils.quat_from_axis_angle(pitch, torch.tensor([0.0, 1.0, 0.0], device=quat.device))
        R_yaw = math_utils.matrix_from_quat(yaw_quat)
        R_roll = math_utils.matrix_from_quat(roll_quat)
        R_pitch = math_utils.matrix_from_quat(pitch_quat)
        R_des = R_yaw.bmm(R_roll).bmm(R_pitch)

        ang_err_mat = 0.5 * (R_des.transpose(-2, -1) @ R - R.transpose(-2, -1) @ R_des)
        ang_err = torch.stack([ang_err_mat[:, 2, 1], ang_err_mat[:, 0, 2], torch.zeros_like(roll.squeeze(-1))], -1)

        body_rate = math_utils.quat_rotate_inverse(quat, ang)
        rate_des = torch.zeros_like(body_rate)
        rate_des[:, 2] = yaw_rate.squeeze(-1)
        rate_err = body_rate - rate_des

        ang_acc = (-ang_err * self._gain_att.to(body_rate) - rate_err * self._gain_rate.to(body_rate) + torch.cross(ang, ang, dim=-1))
        vec = torch.cat([ang_acc, T], -1)
        cmd = (self._mixer @ vec.T).T
        return (cmd / self._max_thrusts) * 2 - 1


class RateController:

    def __init__(self, cfg: RateControllerCfg, num_envs: int, device: str):
        # store inputs
        self.cfg = cfg
        self.num_envs = num_envs
        self._device = device

        self._mixer, self._max_thrusts, _ = compute_mixer_and_limits(cfg.rotor_params, device=device)
        self._mass = torch.tensor(cfg.mass, dtype=torch.float32, device=device)
        self._gain_rate = torch.tensor(cfg.gain_angular_rate, dtype=torch.float32, device=device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the controller's input command."""
        return 4

    """
    Operations.
    """

    def reset(self, env_ids: torch.Tensor = None):
        """Reset the interrnals.

        Args:
            env_ids: The environment indices to reset. If None, then all environments are reset.
        """
        pass

    def compute(
        self,
        root_state: torch.Tensor,
        target_rate: torch.Tensor,
        target_thrust: torch.Tensor,
    ) -> torch.Tensor:
        bs = root_state.shape[:-1]
        cmd = self._compute_impl(root_state.reshape(-1, 13), target_rate.reshape(-1, 3), target_thrust.reshape(-1, 1))
        return cmd.reshape(*bs, -1)

    def _compute_impl(self, state, rate_des, thrust):
        pos, quat, vel, ang = torch.split(state, [3, 4, 3, 3], -1)
        body_rate = math_utils.quat_rotate_inverse(quat, ang)
        rate_err = body_rate - rate_des
        acc_des = -rate_err * self._gain_rate.to(body_rate) + torch.cross(ang, ang, dim=-1)
        vec = torch.cat([acc_des, thrust], -1)
        cmd = (self._mixer @ vec.T).T
        return (cmd / self._max_thrusts) * 2 - 1
