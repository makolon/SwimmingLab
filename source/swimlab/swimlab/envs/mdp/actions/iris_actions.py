from __future__ import annotations
from typing import Sequence, TYPE_CHECKING

import torch
from isaaclab.managers import ActionTerm
from isaaclab.utils import math as math_utils
from swimlab.controllers import LeePositionController

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from . import iris_actions_cfg


class IRISVelocityAction(ActionTerm):
    """Action term converting (v_x,v_y,v_z,yaw) commands into rotor throttles via Lee controller."""

    cfg: iris_actions_cfg.IRISVelocityActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, action_dim)."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: iris_actions_cfg.IRISVelocityActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # rotor dof indices
        rotor_ids, _ = self._asset.find_joints(cfg.joint_names)
        self._rotor_ids = rotor_ids

        # assume first robot asset is the drone
        self._controller = LeePositionController(cfg.controller, num_envs=self.num_envs, device=self.device)

        # action / command buffers
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # scaling tensors
        self._lin_scale = torch.tensor(cfg.linear_scale, device=self.device).reshape(1, 3)
        self._yaw_scale = torch.tensor(cfg.yaw_scale, device=self.device).reshape(1, 1)

        # command storage
        self._target_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_yaw = torch.zeros(self.num_envs, 1, device=self.device)

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

    def apply_actions(self):
        """Convert stored commands to rotor throttles and send to the asset."""
        # assemble root state tensor (pos, quat, lin vel, ang vel)
        root_state = torch.cat(
            [
                self._asset.data.root_pos_w,
                self._asset.data.root_quat_w,
                self._asset.data.root_vel_w[:, :3],  # linear
                self._asset.data.root_vel_w[:, 3:],  # angular
            ],
            dim=-1,
        )
        rotor_cmds = self._controller.compute(
            root_state=root_state,
            target_vel=self._target_vel,
            target_yaw=self._target_yaw,
        )
        rotor_cmd = torch.ones_like(rotor_cmds, device=rotor_cmds.device) * 1e4
        # print("rotor_cmds:", rotor_cmds)

        # fall back to effort interface over rotor joints (slice(None) → all joints)
        self._asset.set_joint_effort_target(rotor_cmds, joint_ids=self._rotor_ids)

    def reset(self, env_ids: Sequence[int] | None = None):
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._target_vel[env_ids] = 0.0
        self._target_yaw[env_ids] = 0.0

