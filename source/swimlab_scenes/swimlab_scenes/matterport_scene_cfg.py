from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg

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


