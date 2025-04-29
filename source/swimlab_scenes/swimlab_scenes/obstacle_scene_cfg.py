import torch

from isaaclab.sensors import ContactSensorCfg, RayCasterCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.terrains import FlatPatchSamplingCfg, HfDiscreteObstaclesTerrainCfg, TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass


##
# Pre-defined config
##
from .base_scene_cfg import BaseSceneCfg  # isort: skip

##
# Scene definition
##
@configclass
class ObstacleSceneCfg(BaseSceneCfg):
    """Configuration for the obstacle avoidance scene with a robot and multiple obstacles.
    """

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=5,
            num_cols=5,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "obstacles": HfDiscreteObstaclesTerrainCfg(
                    size=(8.0, 8.0),
                    horizontal_scale=0.1,
                    vertical_scale=0.1,
                    border_width=0.0,
                    num_obstacles=50,
                    obstacle_height_mode="choice",
                    obstacle_width_range=(0.4, 0.8),
                    obstacle_height_range=(3.0, 6.0),
                    platform_width=1.5,
                    flat_patch_sampling={
                        "target": FlatPatchSamplingCfg(num_patches=5, patch_radius=0.35, max_height_diff=0.05)
                    },
                )
            },
        ),
        max_init_terrain_level=5,
        collision_group=-1,
        debug_vis=False,
    )

    # height scanner
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # lidar
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=False,
        pattern_cfg=patterns.BpearlPatternCfg(
            horizontal_fov=360.0,
            horizontal_res=10.0,
            vertical_ray_angles=torch.linspace(-10, 20, 4).tolist(),
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # contact sensor
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
