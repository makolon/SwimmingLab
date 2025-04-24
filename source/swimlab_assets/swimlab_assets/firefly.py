import isaaclab.sim as sim_utils

from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

from . import SWIMLAB_ASSETS_DATA_DIR


FIREFLY_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{SWIMLAB_ASSETS_DATA_DIR}/Robots/USD/Firefly/firefly.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            linear_damping=0.2,
            angular_damping=0.2,
            enable_gyroscopic_forces=True,
            max_linear_velocity=1000.0,  # m/s
            max_angular_velocity=1000.0,  # rad/s
            max_depenetration_velocity=10.0,
            max_contact_impulse=0.0,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.5),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "thruster": DCMotorCfg(
            joint_names_expr=[
                "rotor_0_joint",
                "rotor_1_joint",
                "rotor_2_joint",
                "rotor_3_joint",
                "rotor_4_joint",
                "rotor_5_joint",
            ],
            saturation_effort=1e8,
            effort_limit=1e8,
            velocity_limit=1e8,
            stiffness={
                "rotor_0_joint": 0.0,
                "rotor_1_joint": 0.0,
                "rotor_2_joint": 0.0,
                "rotor_3_joint": 0.0,
                "rotor_4_joint": 0.0,
                "rotor_5_joint": 0.0,
            },
            damping={
                "rotor_0_joint": 1e6,
                "rotor_1_joint": 1e6,
                "rotor_2_joint": 1e6,
                "rotor_3_joint": 1e6,
                "rotor_4_joint": 1e6,
                "rotor_5_joint": 1e6,
            },
            armature={
                "rotor_0_joint": 0.0,
                "rotor_1_joint": 0.0,
                "rotor_2_joint": 0.0,
                "rotor_3_joint": 0.0,
                "rotor_4_joint": 0.0,
                "rotor_5_joint": 0.0,
            },
            friction={
                "rotor_0_joint": 0.0,
                "rotor_1_joint": 0.0,
                "rotor_2_joint": 0.0,
                "rotor_3_joint": 0.0,
                "rotor_4_joint": 0.0,
                "rotor_5_joint": 0.0,
            },
        ),
    },
)
