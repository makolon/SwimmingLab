
from dataclasses import MISSING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    RigidObjectCfg,
    RigidObjectCollectionCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, ImuCfg, RayCasterCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.terrains import HfDiscreteObstaclesTerrainCfg, TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

##
# Pre-defined config
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

##
# Scene definition
##
@configclass
class ObstacleSceneCfg(InteractiveSceneCfg):
    """Configuration for the sorting scene with a robot and multiple blocks.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING

    # terrain
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     terrain_generator=TerrainGeneratorCfg(
    #         seed=0,
    #         size=(40.0, 40.0),
    #         border_width=5.0,
    #         num_rows=1,
    #         num_cols=1,
    #         horizontal_scale=0.1,
    #         vertical_scale=0.1,
    #         slope_threshold=0.75,
    #         use_cache=False,
    #         color_scheme="height",
    #         sub_terrains={
    #             "obstacles": HfDiscreteObstaclesTerrainCfg(
    #                 horizontal_scale=0.1,
    #                 vertical_scale=0.1,
    #                 border_width=0.0,
    #                 num_obstacles=3000,
    #                 obstacle_height_mode="choice",
    #                 obstacle_width_range=(0.3, 1.0),
    #                 obstacle_height_range=[2.0, 4.0],
    #                 platform_width=1.0,
    #             ),
    #         },
    #     ),
    #     max_init_terrain_level=None,
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     visual_material=sim_utils.MdlFileCfg(
    #         mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
    #         project_uvw=True,
    #         texture_scale=(0.25, 0.25),
    #     ),
    #     debug_vis=False,
    # )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # obstacles
    dynamic_obstacle_collection_cfg = RigidObjectCollectionCfg(
        rigid_objects={
            **{
                f"dynamic_cube_{i}": RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/DynamicObstacleCube_{i}",
                    spawn=sim_utils.CuboidCfg(
                        size=(0.2, 0.2, 3.0),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=True,
                        ),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 1.0, 0.0), metallic=0.2
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(i * 1.0, 0.0, 1.5)),
                )
                for i in range(30)
            },
            **{
                f"dynamic_cylinder_{i}": RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/DynamicObstacleCylinder_{i}",
                    spawn=sim_utils.CylinderCfg(
                        radius=0.1,
                        height=3.0,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=True,
                        ),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.0, 0.0), metallic=0.2
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(i * 1.0, 2.0, 1.5)),
                )
                for i in range(30)
            }
        }
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
            vertical_ray_angles=torch.linspace(-10, 20, 4).tolist(),
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # contact sensor
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0),
    )
