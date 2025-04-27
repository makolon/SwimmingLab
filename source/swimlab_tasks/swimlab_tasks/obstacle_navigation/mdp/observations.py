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
