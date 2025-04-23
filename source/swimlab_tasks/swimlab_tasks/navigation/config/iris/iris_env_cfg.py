from isaaclab.envs.common import ViewerCfg
from isaaclab.utils import configclass

from swimlab.envs.mdp.actions.drone_actions_cfg import DroneVelocityActionCfg
from swimlab.controllers.lee_position_controller_cfg import LeePositionControllerCfg
from swimlab_assets.iris import IRIS_CFG
from swimlab_scenes.base_scene_cfg import BaseSceneCfg
from swimlab_scenes.living_room_scene_cfg import LivingRoomSceneCfg
from swimlab_tasks.navigation.navigation_env_cfg import NavigationBaseEnvCfg


###
# Train
###

@configclass
class IRISPlainNavigationEnvCfg(NavigationBaseEnvCfg):
    """Configuration for the navigation environment."""

    # scene settings
    scene: BaseSceneCfg = BaseSceneCfg(num_envs=4096, env_spacing=10.0)

    # viewer settings
    viewer: ViewerCfg = ViewerCfg(eye=(10.0, 10.0, 10.0), lookat=(0.0, 0.0, 0.0))

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # robot
        self.scene.robot = IRIS_CFG

        self.actions.drone_action = DroneVelocityActionCfg(
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
            controller=LeePositionControllerCfg(
                gravity=9.81,
                position_gain=[4.0, 4.0, 4.0],
                velocity_gain=[2.2, 2.2, 2.2],
                attitude_gain=[0.7, 0.7, 0.035],
                angular_rate_gain=[0.1, 0.1, 0.025],
            ),
            rotor_params={
                "num_rotors": 4,
                "rotor_angles": [-0.533708, 2.565218, 0.533708, -2.565218],
                "arm_lengths": [0.255539, 0.238537, 0.238539, 0.238537],
                "force_constants": [8.54858e-06, 8.54858e-06, 8.54858e-06, 8.54858e-06],
                "moment_constants": [1.3677728816219314e-07, 1.3677728816219314e-07, 1.3677728816219314e-07, 1.3677728816219314e-07],
                "kf": [6.11-8] * 4,
                "km": [1.5e-9] * 4,
                "directions": [1.0, -1.0, 1.0, -1.0],
                "max_rotation_velocities": [838, 838, 838, 838],
                "mass": 1.52,
                "inertia": {"xx": 0.0347563, "xy": 0.0, "xz": 0.0, "yy": 0.0458929, "yz": 0.0, "zz": 0.0977},
                "drag_coef": 0.1,
                "tau_up": [0.43] * 4,
                "tau_down": [0.43] * 4,
            }
        )


@configclass
class IRISLivingRoomNavigationEnvCfg(NavigationBaseEnvCfg):
    """Configuration for the tidyup environment."""

    # scene settings
    scene: LivingRoomSceneCfg = LivingRoomSceneCfg(num_envs=4096, env_spacing=2.5)

    # viewer settings
    viewer: ViewerCfg = ViewerCfg(eye=(-2.0, -3.5, 1.2), lookat=(4.0, 3.0, 0.0))

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # robot
        self.scene.robot = IRIS_CFG

        self.actions.drone_action = mdp.DroneVelocityActionCfg(
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
            controller=LeePositionControllerCfg(
                gravity=9.81,
                position_gain=[4.0, 4.0, 4.0],
                velocity_gain=[2.2, 2.2, 2.2],
                attitude_gain=[0.7, 0.7, 0.035],
                angular_rate_gain=[0.1, 0.1, 0.025],
            ),
            rotor_params={
                "num_rotors": 4,
                "rotor_angles": [-0.533708, 2.565218, 0.533708, -2.565218],
                "arm_lengths": [0.255539, 0.238537, 0.238539, 0.238537],
                "force_constants": [8.54858e-06, 8.54858e-06, 8.54858e-06, 8.54858e-06],
                "moment_constants": [1.3677728816219314e-07, 1.3677728816219314e-07, 1.3677728816219314e-07, 1.3677728816219314e-07],
                "kf": [6.11-8] * 4,
                "km": [1.5e-9] * 4,
                "directions": [1.0, -1.0, 1.0, -1.0],
                "max_rotation_velocities": [838, 838, 838, 838],
                "mass": 1.52,
                "inertia": {"xx": 0.0347563, "xy": 0.0, "xz": 0.0, "yy": 0.0458929, "yz": 0.0, "zz": 0.0977},
                "drag_coef": 0.1,
                "tau_up": [0.43] * 4,
                "tau_down": [0.43] * 4,
            }
        )


###
# Play
###

@configclass
class IRISPlainNavigationEnvCfg_PLAY(IRISPlainNavigationEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 10.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class IRISLivingRoomNavigationEnvCfg_PLAY(IRISLivingRoomNavigationEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 10.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False

