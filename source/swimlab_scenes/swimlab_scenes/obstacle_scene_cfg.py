import isaaclab.sim as sim_utils
from isaaclab.assets import (
    RigidObjectCfg,
    RigidObjectCollectionCfg,
)
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


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

