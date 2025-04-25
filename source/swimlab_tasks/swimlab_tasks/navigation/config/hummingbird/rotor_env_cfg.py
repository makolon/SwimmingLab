
from isaaclab.envs.common import ViewerCfg
from isaaclab.utils import configclass

from swimlab.envs.mdp.actions.rotor_actions_cfg import RotorActionCfg
from swimlab_assets.hummingbird import HUMMINGBIRD_CFG
from swimlab_scenes.base_scene_cfg import BaseSceneCfg
from swimlab_scenes.living_room_scene_cfg import LivingRoomSceneCfg
from swimlab_tasks.navigation.navigation_env_cfg import NavigationBaseEnvCfg


###
# Train
###

@configclass
class HummingBirdPlaneNavigationEnvCfg(NavigationBaseEnvCfg):
    """Configuration for the navigation environment."""

    # scene settings
    scene: BaseSceneCfg = BaseSceneCfg(num_envs=4096, env_spacing=3.0)

    # viewer settings
    viewer: ViewerCfg = ViewerCfg(eye=(10.0, 10.0, 10.0), lookat=(0.0, 0.0, 0.0))

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # robot
        self.scene.robot = HUMMINGBIRD_CFG

        self.actions.drone_action = RotorActionCfg(
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
            rotor_params={
                "num_rotors": 4,
                "rotor_angles": [0.0, 1.57079632679, 3.14159265359, -1.57079632679],
                "arm_lengths": [0.17, 0.17, 0.17, 0.17],
                "force_constants": [8.54858e-06, 8.54858e-06, 8.54858e-06, 8.54858e-06],
                "moment_constants": [1.3677728816219314e-07, 1.3677728816219314e-07, 1.3677728816219314e-07, 1.3677728816219314e-07],
                "kf": [6.003189] * 4,
                "km": [0.001053366] * 4,
                "directions": [-1.0, 1.0, -1.0, 1.0],
                "max_rotation_velocities": [838, 838, 838, 838],
                "mass": 0.716,
                "inertia": {"xx": 0.07, "xy": 0.0, "xz": 0.0, "yy": 0.07, "yz": 0.0, "zz": 0.012},
                "drag_coef": 0.2,
                "tau_up": [0.43] * 4,
                "tau_down": [0.43] * 4,
            }
        )


@configclass
class HummingBirdLivingRoomNavigationEnvCfg(NavigationBaseEnvCfg):
    """Configuration for the tidyup environment."""

    # scene settings
    scene: LivingRoomSceneCfg = LivingRoomSceneCfg(num_envs=4096, env_spacing=2.5)

    # viewer settings
    viewer: ViewerCfg = ViewerCfg(eye=(10.0, 10.0, 10.0), lookat=(0.0, 0.0, 0.0))

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # robot
        self.scene.robot = HUMMINGBIRD_CFG

        self.actions.drone_action = RotorActionCfg(
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
            rotor_params={
                "num_rotors": 4,
                "rotor_angles": [0.0, 1.57079632679, 3.14159265359, -1.57079632679],
                "arm_lengths": [0.17, 0.17, 0.17, 0.17],
                "force_constants": [8.54858e-06, 8.54858e-06, 8.54858e-06, 8.54858e-06],
                "moment_constants": [1.3677728816219314e-07, 1.3677728816219314e-07, 1.3677728816219314e-07, 1.3677728816219314e-07],
                "kf": [6.003189] * 4,
                "km": [0.001053366] * 4,
                "directions": [-1.0, 1.0, -1.0, 1.0],
                "max_rotation_velocities": [838, 838, 838, 838],
                "mass": 0.716,
                "inertia": {"xx": 0.07, "xy": 0.0, "xz": 0.0, "yy": 0.07, "yz": 0.0, "zz": 0.012},
                "drag_coef": 0.2,
                "tau_up": [0.43] * 6,
                "tau_down": [0.43] * 6,
            }
        )


###
# Play
###

@configclass
class HummingBirdPlaneNavigationEnvCfg_PLAY(HummingBirdPlaneNavigationEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 10.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class HummingBirdLivingRoomNavigationEnvCfg_PLAY(HummingBirdLivingRoomNavigationEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 10.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False

