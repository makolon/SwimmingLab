from __future__ import annotations
from typing import Sequence, TYPE_CHECKING

import torch
from isaaclab.managers import ActionTerm
from isaaclab.utils import math as math_utils
from swimlab.controllers import LeeController

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from . import lee_actions_cfg


class LeeAction(ActionTerm):
    """Lee-controller position, velocity, acceleration (+yaw) action to rotor forces/thrust for a multirotor."""

    cfg: lee_actions_cfg.LeeActionCfg
    _asset: "Articulation"  # injected by ActionTerm

    def __init__(self, cfg: lee_actions_cfg.LeeActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # joint / body indices
        self._rotor_joint_ids, _ = self._asset.find_joints(cfg.rotor_joint_names)
        self._rotor_body_ids, _ = self._asset.find_bodies(cfg.rotor_body_names)
        self._base_body_ids, _ = self._asset.find_bodies(cfg.base_body_names)

        # controller
        self._controller = LeeController(
            cfg.controller, cfg.rotor_params, num_envs=self.num_envs, device=self.device
        )

        # action buffers
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # scaling factor
        self._lin_scale = torch.tensor(cfg.linear_scale, device=self.device).view(1, 3)
        self._yaw_scale = torch.tensor(cfg.yaw_scale, device=self.device).view(1, 1)

        # command storage
        self._target_cmd = torch.zeros(self.num_envs, 4, device=self.device)

        # rotor parameters
        self._num_rotors = torch.as_tensor(cfg.rotor_params["num_rotors"], device=self.device).view(1, -1)
        self._tau_up = torch.as_tensor(cfg.rotor_params["tau_up"], device=self.device).view(1, -1)
        self._tau_down = torch.as_tensor(cfg.rotor_params["tau_down"], device=self.device).view(1, -1)
        self._kf = torch.as_tensor(cfg.rotor_params["kf"], device=self.device).view(1, -1)
        self._km = torch.as_tensor(cfg.rotor_params["km"], device=self.device).view(1, -1)
        self._directions = torch.as_tensor(cfg.rotor_params["directions"], device=self.device).view(1, -1)
        self._max_rotvel = torch.as_tensor(cfg.rotor_params["max_rotation_velocities"], device=self.device).view(1, -1)
        self._drag_coef = torch.as_tensor(cfg.rotor_params["drag_coef"], device=self.device).view(1, -1)
        self._mass = torch.as_tensor(cfg.rotor_params["mass"], device=self.device).view(1, -1)

        # run-time buffers
        self._rotor_throttle = torch.zeros(self.num_envs, self._num_rotors, device=self.device)
        self._rotor_velocities = torch.zeros(self.num_envs, self._num_rotors, device=self.device)
        self._rotor_thrusts = torch.zeros(self.num_envs, self._num_rotors, 3, device=self.device)
        self._body_torques = torch.zeros(self.num_envs, 3, device=self.device)
        self._body_forces = torch.zeros(self.num_envs, 3, device=self.device)

        # constant world-frame Z axis for torque computation
        self._z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)

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
        self._raw_actions[:] = actions
        self._processed_actions[:, :3] = actions[:, :3] * self._lin_scale
        self._processed_actions[:, 3:4] = actions[:, 3:4] * self._yaw_scale
        if self.cfg.clip is not None:
            self._processed_actions.clamp_(-self.cfg.clip, self.cfg.clip)
        self._target_cmd[:] = self._processed_actions[:, :4]

    def apply_actions(self) -> torch.Tensor:
        root_state = torch.cat(
            [
                self._asset.data.root_pos_w,
                self._asset.data.root_quat_w,
                self._asset.data.root_vel_w[:, :3],
                self._asset.data.root_vel_w[:, 3:],
            ],
            dim=-1,
        )

        rotor_cmd = self._controller.compute(
            root_state=root_state,
            target_cmd=self._target_cmd
        ).clamp_(-1.0, 1.0)

        target_thr = torch.sqrt(torch.clamp((rotor_cmd + 1.0) * 0.5, 0.0, 1.0))
        tau = torch.where(target_thr > self._rotor_throttle, self._tau_up, self._tau_down).clamp_(0.0, 1.0)
        self._rotor_throttle += tau * (target_thr - self._rotor_throttle)

        t = torch.square(self._rotor_throttle).clamp_(0.0, 1.0)
        thrusts = t * self._kf
        moments = (t * self._km) * -self._directions

        body_state = self._asset.data.body_state_w[:, self._rotor_body_ids, :]
        rotor_rot_w = body_state[..., 3:7]

        torque_axis = math_utils.quat_rotate(
            rotor_rot_w.flatten(end_dim=-2),
            self._z_axis.expand(self.num_envs, self._num_rotors, 3).flatten(end_dim=-2),
        ).unflatten(0, (self.num_envs, self._num_rotors))

        self._rotor_thrusts.zero_()
        self._rotor_thrusts[..., 2] = thrusts
        self._body_torques.zero_()
        self._body_torques[:] = (moments.unsqueeze(-1) * torque_axis).sum(-2)

        # drag force
        body_vel = self._asset.data.root_vel_w[:, :]
        self._body_forces.zero_()
        self._body_forces[:] += (self._drag_coef * self._mass) * body_vel[..., :3]

        # apply per-rotor forces
        self._asset.set_external_force_and_torque(
            forces=self._rotor_thrusts,
            torques=torch.zeros_like(self._rotor_thrusts),
            body_ids=self._rotor_body_ids,
        )

        # apply aggregate force / torque on base link(s)
        B = len(self._base_body_ids)
        self._asset.set_external_force_and_torque(
            forces=self._body_forces.unsqueeze(1).expand(-1, B, -1),
            torques=self._body_torques.unsqueeze(1).expand(-1, B, -1),
            body_ids=self._base_body_ids,
        )

        # spin joints for visualisation
        self._rotor_velocities = self._rotor_throttle * self._directions * self._max_rotvel
        self._asset.write_joint_velocity_to_sim(
            velocity=self._rotor_velocities,
            joint_ids=self._rotor_joint_ids,
        )

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._target_cmd[env_ids] = 0.0
        self._rotor_throttle[env_ids] = 0.0
        self._rotor_velocities[env_ids] = 0.0
        self._rotor_thrusts[env_ids] = 0.0
        self._body_forces[env_ids] = 0.0
        self._body_torques[env_ids] = 0.0

