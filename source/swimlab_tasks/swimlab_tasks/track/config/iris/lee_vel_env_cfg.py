from isaaclab.envs.common import ViewerCfg
from isaaclab.utils import configclass

from swimlab.envs.mdp.actions.lee_actions_cfg import LeeActionCfg
from swimlab.controllers.lee_controller_cfg import LeeControllerCfg
from swimlab_assets.iris import IRIS_CFG
from swimlab_scenes.base_scene_cfg import BaseSceneCfg
from swimlab_tasks.track.track_env_cfg import TrackBaseEnvCfg


###
# Train
###

@configclass
class IRISPlaneTrackEnvCfg(TrackBaseEnvCfg):
    """Configuration for the track environment."""

    # scene settings
    scene: BaseSceneCfg = BaseSceneCfg(num_envs=4096, env_spacing=3.0)

    # viewer settings
    viewer: ViewerCfg = ViewerCfg(eye=(10.0, 10.0, 10.0), lookat=(0.0, 0.0, 0.0))

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # robot
        self.scene.robot = IRIS_CFG

        self.actions.drone_action = LeeActionCfg(
            asset_name="robot",
            rotor_joint_names=[
                "rotor_0_joint",
                "rotor_1_joint",
                "rotor_2_joint",
                "rotor_3_joint",
            ],
            rotor_body_names=[
                "rotor_0",
                "rotor_1",
                "rotor_2",
                "rotor_3",
            ],
            base_body_names=["base_link"],
            linear_scale=[1.0, 1.0, 1.0],
            yaw_scale=1.0,
            controller=LeeControllerCfg(
                controller_type="velocity",
                gravity=9.81,
                position_gain=[6.0, 6.0, 6.0],
                velocity_gain=[4.7, 4.7, 4.7],
                attitude_gain=[3.0, 3.0, 0.15],
                angular_rate_gain=[0.52, 0.52, 0.18],
            ),
            rotor_params={
                "num_rotors": 4,
                "rotor_angles": [-0.533708, 2.565218, 0.533708, -2.565218],
                "arm_lengths": [0.255539, 0.238537, 0.238539, 0.238537],
                "force_constants": [8.54858e-06, 8.54858e-06, 8.54858e-06, 8.54858e-06],
                "moment_constants": [1.3677728816219314e-07, 1.3677728816219314e-07, 1.3677728816219314e-07, 1.3677728816219314e-07],
                "kf": [6.003189] * 4,
                "km": [0.001053366] * 4,
                "directions": [1.0, 1.0, -1.0, -1.0],
                "max_rotation_velocities": [838, 838, 838, 838],
                "mass": 1.52,
                "inertia": {"xx": 0.0347563, "xy": 0.0, "xz": 0.0, "yy": 0.0458929, "yz": 0.0, "zz": 0.0977},
                "drag_coef": 0.2,
                "tau_up": [0.43] * 4,
                "tau_down": [0.43] * 4,
            }
        )


###
# Play
###

@configclass
class IRISPlaneTrackEnvCfg_PLAY(IRISPlaneTrackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 10.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False

