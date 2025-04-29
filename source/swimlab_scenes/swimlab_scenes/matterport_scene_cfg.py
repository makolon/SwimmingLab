import torch

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg, ContactSensorCfg, RayCasterCfg
from isaaclab.sensors.ray_caster import patterns

##
# Pre-defined configuration
##
from .base_scene_cfg import BaseSceneCfg

##
# Scene definition
##
@configclass
class MatterportSceneCfg(BaseSceneCfg):
    """Configuration for the living room scene with a robot and multiple objects.
    """

    # camera
    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/Camera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focus_distance=400.0,
            focal_length=1.66,  # NOTE: (640/2) / tan(1.047/2)
            horizontal_aperture=1.89,  # NOTE: 640 * 0.003
            vertical_aperture=1.44,  # NOTE: 480 * 0.003
            clipping_range=(0.01, 1.0e3),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.1),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
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
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    # contact sensor
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

