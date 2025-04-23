from __future__ import annotations
from typing import Sequence, TYPE_CHECKING

import torch
from isaaclab.managers import ActionTerm
from isaaclab.utils import math as math_utils
from swimlab.controllers import LeePositionController

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from . import actions_cfg


class DroneVelocityAction(ActionTerm):
    """Action term converting (v_x,v_y,v_z,yaw) commands into rotor throttles via Lee controller."""

    cfg: actions_cfg.DroneVelocityActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, action_dim)."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: actions_cfg.DroneVelocityActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # rotor dof indices
        rotor_joint_ids, _ = self._asset.find_joints(cfg.rotor_joint_names)
        self._rotor_joint_ids = rotor_joint_ids

        # rotor body indices
        rotor_body_ids, _ = self._asset.find_bodies(cfg.rotor_body_names)
        self._rotor_body_ids = rotor_body_ids

        # base body indices
        base_body_ids, _ = self._asset.find_bodies(cfg.base_body_names)
        self._base_bdoy_ids = base_body_ids

        # assume first robot asset is the drone
        self._controller = LeePositionController(cfg.controller, cfg.rotor_params, num_envs=self.num_envs, device=self.device)

        # action / command buffers
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # rotor parameters
        self._lin_scale = torch.tensor(cfg.linear_scale, device=self.device).reshape(1, 3)
        self._yaw_scale = torch.tensor(cfg.yaw_scale, device=self.device).reshape(1, 1)

        # command storage
        self._target_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_yaw = torch.zeros(self.num_envs, 1, device=self.device)

        # create buffers
        self.rotor_velocities = torch.zeros(self.num_envs, self.cfg.rotor_params.num_rotors, device=self.device)
        self.rotor_thrusts = torch.zeros(self.num_envs, self.cfg.rotor_params.num_rotors, 3, device=self.device)
        self.body_torques = torch.zeros(self.num_envs, 3, device=self.device)
        self.body_forces = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Scale and store high‑level commands."""
        self._raw_actions[:] = actions

        # scale
        self._processed_actions[:, :3] = actions[:, :3] * self._lin_scale
        self._processed_actions[:, 3:4] = actions[:, 3:4] * self._yaw_scale

        # optional clip
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(self._processed_actions, -self.cfg.clip, self.cfg.clip)

        # save targets for apply stage
        self._target_vel[:] = self._processed_actions[:, :3]
        self._target_yaw[:] = self._processed_actions[:, 3:4]

    def apply_actions(self) -> torch.Tensor:
        """
        1. Convert the stored (v_x, v_y, v_z, yaw) commands to rotor throttle.
        2. Transform throttle to thrust & reaction-moment (using self._rotor_model).
        3. Compose:
           - per-rotor force vectors > applied to rotor rigid bodies
           - aggregate external force > applied to body link (drag, down-wash)
           - aggregate external torque
        4. Push all wrenches with `set_external_force_and_torque`.
        """
        root_state = torch.cat(
            [
                self._asset.data.root_pos_w,
                self._asset.data.root_quat_w,
                self._asset.data.root_vel_w[:, :3],  # linear
                self._asset.data.root_vel_w[:, 3:],  # angular
            ],
            dim=-1,
        )
        # Controller output: throttle in [0, 1] per rotor
        rotor_cmd = self._controller.compute(
            root_state=root_state,
            target_vel=self._target_vel,
            target_yaw=self._target_yaw,
        ).clamp_(0.0, 1.0)  # (N, R)

        target_throttle = torch.sqrt(torch.clamp((rotor_cmd + 1) / 2, 0, 1))
        tau = torch.where(target_throttle > self._rotor_throttle, self.cfg.rotor_params.tau_up, self.cfg.rotor_params.tau_down)
        tau = torch.clamp(tau, 0, 1)

        self._rotor_throttle.add_(tau * (target_throttle - self._rotor_throttle))  # TODO: Fix this
        t = torch.clamp(torch.square(self._rotor_throttle), 0.0, 1.0)
        thrusts = t * self.cfg.rotor_params.kf
        moments = (t * self.cfg.rotor_params.km) * -self.cfg.rotor_params.directions

        # Rotor world poses (centre-of-mass of each rotor rigid body)
        rotor_pos_w, rotor_rot_w = self._asset.data.body_state_w[:, self._rotor_body_ids, :]
        torque_axis = math_utils.quat_axis(
            rotor_rot_w.flatten(end_dim=-2), axis=2
        ).unflatten(0, (self.num_envs, self.cfg.rotor_params.num_rotors))

        # per-rotor force vectors (body-frame +Z)
        self._rotor_thrusts = torch.zeros((self.num_envs, self.cfg.rotor_params.num_rotors, 3), device=self.device)
        self._rotor_thrusts[..., 2] = thrusts

        # calculate body torque
        self._body_torques = (moments.unsqueeze(-1) * torque_axis).sum(-2)

        # external aerodynamic forces (drag + downwash)
        self._body_forces.zero_()

        body_pos, body_rot = self._asset.data.body_pos_w, self._asset.data.body_quat_w
        body_vel = self._asset.data.body_vel_w
        self._body_forces += vmap(self.downwash)(
            body_pos,
            body_pos,
            math_utils.quat_rotate(body_rot, self._rotor_thrusts.sum(-2)),
            kz=0.3,
        ).sum(-2)
        self._body_forces += (self.cfg.rotor_params.drag_coef * self.cfg.rotor_params.mass) * body_vel[..., :3]

        # Apply rotor-level forces (local frame)
        self._asset.set_external_force_and_torque(
            forces=self._rotor_thrusts,
            torques=torch.zeros_like(self._rotor_thrusts),
            body_ids=self._rotor_body_ids,
        )

        # Apply aggregate forces/torques to the body link
        self._asset.set_external_force_and_torque(
            forces=self._body_forces,
            torques=self._body_torques,
            body_ids=self._base_body_ids,

        # spin rotor joints
        self._rotor_velocities = rotor_cmd * self.cfg.rotor_params.directions * self.cfg.rotor_params.max_rotation_velocities
        self._asset.set_joint_velocity_target(
            target=self._rotor_velocities,
            joint_indices=self._rotor_joint_ids,
        )

    @staticmethod
    def downwash(
        p0: torch.Tensor,
        p1: torch.Tensor,
        p1_t: torch.Tensor,
        kr: float=2,
        kz: float=1,
    ):
        """
        A highly simplified downwash effect model.

        References:
        https://arxiv.org/pdf/2207.09645.pdf
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8798116

        """
        z, r = separation(p0, p1, normalize(p1_t))
        z = torch.clip(z, 0)
        v = torch.exp(-0.5 * torch.square(kr * r / z)) / (1 + kz * z)**2
        f = off_diag(v * - p1_t)
        return f

    def reset(self, env_ids: Sequence[int] | None = None):
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._target_vel[env_ids] = 0.0
        self._target_yaw[env_ids] = 0.0
        self._rotor_velocities[env_ids] = 0.0
        self._rotor_thrusts[env_ids] = 0.0
        self._body_forces[env_ids] = 0.0
        self._body_torques[env_ids] = 0.0
