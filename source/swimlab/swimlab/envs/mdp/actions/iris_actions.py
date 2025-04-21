from __future__ import annotations
from typing import Sequence, TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm
from isaaclab.utils.math import quat_rotate_inverse

if TYPE_CHECKING:  # Forward‑decls for type checkers only
    from isaaclab.envs import ManagerBasedEnv
    from . import actions_cfg


def _build_mixer(rotor_cfg: dict, inertia_diag: torch.Tensor) -> torch.Tensor:
    """Return the 4×4 mixer matrix mapping [τx, τy, τz, F]ᵀ → rotor thrusts."""
    angles = torch.as_tensor(rotor_cfg["rotor_angles"])
    arms = torch.as_tensor(rotor_cfg["arm_lengths"])
    k_f = torch.as_tensor(rotor_cfg["force_constants"])
    k_m = torch.as_tensor(rotor_cfg["moment_constants"])
    dirs = torch.as_tensor(rotor_cfg["directions"])

    a = torch.stack(
        [
            torch.sin(angles) * arms,
            -torch.cos(angles) * arms,
            -dirs * k_m / k_f,
            torch.ones_like(angles),
        ]
    )
    return a.T @ (a @ a.T).inverse() @ inertia_diag


class IRISVelocityAction(ActionTerm):
    """
    Converts a normalized 6‑D body‑frame velocity command
    (vx, vy, vz, roll‑rate, pitch‑rate, yaw‑rate) ∈ [-1, 1]
    into four normalized rotor speed commands for the IRIS quadrotor.
    """

    cfg: actions_cfg.IRISVelocityActionCfg
    _asset: Articulation
    _scale: torch.Tensor
    _offset: torch.Tensor
    _bounding_strategy: str | None

    def __init__(self, cfg: actions_cfg.IRISVelocityActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Rotor DOF indices
        rotor_ids, _ = self._asset.find_joints(cfg.rotor_joint_names)
        self._rotor_ids = rotor_ids

        # Physical parameters
        self._mass = torch.tensor(cfg.mass, device=self.device)
        self._inertia = torch.diag(torch.tensor(cfg.inertia, device=self.device))
        self._gravity = torch.tensor([0.0, 0.0, 9.81], device=self.device)

        # Mixer matrix and thrust limits
        inertia4 = torch.diag_embed(torch.cat((self._inertia.diag(), torch.ones(1, device=self.device))))
        mixer = _build_mixer(cfg.rotor_config, inertia4).to(self.device)
        self.register_buffer("_mixer", mixer, persistent=False)

        k_f = torch.as_tensor(cfg.rotor_config["force_constants"], device=self.device)
        w_max_sq = torch.as_tensor(cfg.rotor_config["max_rotation_velocities"], device=self.device).square()
        self.register_buffer("_thr_max", w_max_sq * k_f, persistent=False)

        # Action scaling/offset (optional; keep unity by default)
        self._scale = torch.tensor(cfg.scale, device=self.device, dtype=torch.float32)
        self._offset = torch.tensor(cfg.offset, device=self.device, dtype=torch.float32)
        self._bounding_strategy = cfg.bounding_strategy

        self._v_max = cfg.v_max
        self._w_max = cfg.w_max

        # Buffers for actions
        self._raw_actions = torch.zeros(env.num_envs, self.action_dim, device=self.device)

    # Properties
    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    # Operations
    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        if self._bounding_strategy == "clip":
            bounded = torch.clip(actions, -1.0, 1.0)
        elif self._bounding_strategy == "tanh":
            bounded = torch.tanh(actions)
        else:
            bounded = actions
        self._processed_actions = bounded * self._scale + self._offset

    def apply_actions(self):
        # Split processed actions
        vel_cmd = self._processed_actions[:, :3] * self._v_max
        rate_cmd = self._processed_actions[:, 3:] * self._w_max

        # Current root state
        root_state = self._asset.data.root_state
        quat = root_state[:, 3:7]
        lin_vel = root_state[:, 7:10]
        ang_vel = root_state[:, 10:13]

        # Transform to body frame
        vel_body = quat_rotate_inverse(quat, lin_vel)
        ang_body = quat_rotate_inverse(quat, ang_vel)

        # Simple proportional control gains (tune as needed)
        vel_err = vel_cmd - vel_body
        rate_err = rate_cmd - ang_body

        force_b = self._mass * (2.0 * vel_err + self._gravity)
        torque_b = self._inertia @ rate_err.unsqueeze(-1)

        in_vec = torch.cat([torque_b.squeeze(-1), force_b[:, 2:3]], dim=-1)
        thrusts = (self._mixer @ in_vec.T).T

        thrusts = torch.clamp(thrusts, 0.0, self._thr_max)
        rotor_cmd = thrusts / self._thr_max * 2.0 - 1.0  # map to [-1, 1]

        self._asset.set_joint_velocity_target(rotor_cmd, joint_ids=self._rotor_ids)

