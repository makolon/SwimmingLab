"""Functions specific to the obstacle navigation environments."""

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

    # lidar parameters
    lidar_range: torch.Tensor = torch.tensor(lidar_range, device=env.device)
    lidar_resolution: torch.Tensor = torch.tensor(lidar_resolution, device=env.device)

    # lidar scan: scan = range - (hit_points - sensoe_pos)
    return sensor.data.ray_hits_w[..., 0] - sensor.data.pos_w[:, 0].unsqueeze(1)
