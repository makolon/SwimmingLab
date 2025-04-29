"""Functions specific to the obstacle navigation environments."""

import numpy as np
import torch

from isaaclab.sensors import RayCaster
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def lidar_scan(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, lidar_range: list, lidar_resolution: float) -> torch.Tensor:
    """Lidar scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    scan_shape = np.prod(lidar_resolution)
    scan_data = lidar_range - (
        (sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)).norm(dim=-1).clamp_max(lidar_range).reshape(-1, scan_shape)
    )
    # lidar scan: scan = range - (hit_points - sensoe_pos)
    return scan_data


def depth_scan(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, data_type: str) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    sensor_data = sensor.data.output[data_type]

    return sensor_data.view(-1, sensor_data.shape[1] * sensor_data.shape[2])


def relative_to_target(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return a boolean tensor indicating whether the target is reached (distance below threshold).

    Args:
        env (ManagerBasedRLEnv): the environment instance.
        command_name (str): name of the command to get target position.

    Returns:
        torch.Tensor: A boolean tensor (shape: (num_envs,)) indicating if the target is reached.
    """
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, 1:2]

    asset = env.scene[asset_cfg.name]
    asset_pos_b = asset.data.root_pos_w[:, 1:2]

    distance = des_pos_b - asset_pos_b
    return distance