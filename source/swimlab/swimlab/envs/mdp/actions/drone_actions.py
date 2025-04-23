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
        self._base_body_ids = base_body_ids

        # assume first robot asset is the drone
        self._controller = LeePositionController(cfg.controller, cfg.rotor_params, num_envs=self.num_envs, device=self.device)

        # action / command buffers
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # action scale parameters
        self._lin_scale = torch.tensor(cfg.linear_scale, device=self.device).reshape(1, 3)
        self._yaw_scale = torch.tensor(cfg.yaw_scale, device=self.device).reshape(1, 1)

        # command storage
        self._target_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_yaw = torch.zeros(self.num_envs, 1, device=self.device)

        # rotor params
        self._num_rotors = torch.tensor(self.cfg.rotor_params["num_rotors"], device=self.device)
        self._tau_up = torch.tensor(self.cfg.rotor_params["tau_up"], device=self.device)
        self._tau_down = torch.tensor(self.cfg.rotor_params["tau_down"], device=self.device)
        self._kf = torch.tensor(self.cfg.rotor_params["kf"], device=self.device)
        self._km = torch.tensor(self.cfg.rotor_params["km"], device=self.device)
        self._directions = torch.tensor(self.cfg.rotor_params["directions"], device=self.device)
        self._drag_coef = torch.tensor(self.cfg.rotor_params["drag_coef"], device=self.device)
        self._mass = torch.tensor(self.cfg.rotor_params["mass"], device=self.device)
        self._max_rotation_velocities = torch.tensor(self.cfg.rotor_params["max_rotation_velocities"], device=self.device)

        # create buffers
        self._rotor_throttle = torch.zeros(self.num_envs, self._num_rotors, device=self.device)
        self._rotor_velocities = torch.zeros(self.num_envs, self._num_rotors, device=self.device)
        self._rotor_thrusts = torch.zeros(self.num_envs, self._num_rotors, 3, device=self.device)
        self._body_torques = torch.zeros(self.num_envs, 3, device=self.device)
        self._body_forces = torch.zeros(self.num_envs, 3, device=self.device)

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
        ).clamp_(-1.0, 1.0)  # (N, R)

        target_throttle = torch.sqrt(torch.clamp((rotor_cmd + 1.0) * 0.5, 0.0, 1.0))
        tau = torch.where(
            target_throttle > self._rotor_throttle,
            self._tau_up,
            self._tau_down,
        ).clamp_(0.0, 1.0)
        self._rotor_throttle.add_(tau * (target_throttle - self._rotor_throttle))

        t = torch.clamp(torch.square(self._rotor_throttle), 0.0, 1.0)
        thrusts = t * self._kf
        moments = (t * self._km) * -self._directions

        # Rotor world poses (centre-of-mass of each rotor rigid body)
        body_state = self._asset.data.body_state_w[:, self._rotor_body_ids, :]
        rotor_pos_w  = body_state[..., 0:3]
        rotor_rot_w  = body_state[..., 3:7]

        z_axis_local = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=rotor_rot_w.dtype)
        z_axis_local = z_axis_local.expand_as(rotor_pos_w)
        torque_axis = math_utils.quat_rotate(
            rotor_rot_w.flatten(end_dim=-2),
            z_axis_local.flatten(end_dim=-2)
        ).unflatten(0, (self.num_envs, self._num_rotors))

        # per-rotor force vectors (body-frame +Z)
        self._rotor_thrusts = torch.zeros((self.num_envs, self._num_rotors, 3), device=self.device)
        self._rotor_thrusts[..., 2] = thrusts

        # calculate body torque
        self._body_torques = (moments.unsqueeze(-1) * torque_axis).sum(-2)

        # external aerodynamic forces (drag + downwash)
        self._body_forces.zero_()

        body_vel = self._asset.data.body_vel_w[:, self._base_body_ids, :]
        drag = (self._drag_coef * self._mass) * body_vel[..., :3]
        self._body_forces += drag.sum(dim=1)
        # self._body_forces += (self._drag_coef * self._mass) * body_vel[..., :3]

        # Apply rotor-level forces (local frame)
        self._asset.set_external_force_and_torque(
            forces=self._rotor_thrusts,
            torques=torch.zeros_like(self._rotor_thrusts),
            body_ids=self._rotor_body_ids,
        )

        # Apply aggregate forces/torques to the body link
        self._asset.set_external_force_and_torque(
            forces=self._body_forces.unsqueeze(1),  # TODO: Fix this
            torques=self._body_torques.unsqueeze(1),  # TODO: Fix this
            body_ids=self._base_body_ids,
        )

        # spin rotor joints
        self._rotor_velocities = (
            self._rotor_throttle
            * self._directions
            * self._max_rotation_velocities
        )
        self._asset.set_joint_velocity_target(
            target=self._rotor_velocities,
            joint_ids=self._rotor_joint_ids,
        )

    def reset(self, env_ids: Sequence[int] | None = None):
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._target_vel[env_ids] = 0.0
        self._target_yaw[env_ids] = 0.0
        self._rotor_velocities[env_ids] = 0.0
        self._rotor_thrusts[env_ids] = 0.0
        self._body_forces[env_ids] = 0.0
        self._body_torques[env_ids] = 0.0
