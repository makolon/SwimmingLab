from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObjectCollection
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_obstacle_pose_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position uniformly within the given ranges.

    This function randomizes the root position of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObjectCollection = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_object_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), asset.num_objects, 6), device=asset.device)

    positions = root_states[:, :, 0:3] + rand_samples[:, :, 0:3]
    positions = positions + env.scene.env_origins[env_ids][:, None, :]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, :, 3], rand_samples[:, :, 4], rand_samples[:, :, 5])
    orientations = math_utils.quat_mul(root_states[:, :, 3:7], orientations_delta)

    # set into the physics simulation
    asset.write_object_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
