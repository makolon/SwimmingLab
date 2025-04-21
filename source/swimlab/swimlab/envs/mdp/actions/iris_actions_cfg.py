from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass
from . import iris_actions


# Rotor parameters for the IRIS quadrotor
ROTOR_CONFIG = {
    "rotor_angles": [0.0, 1.5708, 3.1416, -1.5708],
    "arm_lengths": [0.17, 0.17, 0.17, 0.17],
    "force_constants": [8.54858e-6] * 4,
    "moment_constants": [1.6e-2] * 4,
    "directions": [1, -1, 1, -1],
    "max_rotation_velocities": [838.0] * 4,
}


@configclass
class IRISVelocityActionCfg(ActionTermCfg):
    """6‑DoF velocity → 4 rotor commands for the IRIS quadrotor."""
    class_type: type = iris_actions.IRISVelocityAction

    rotor_joint_names: list[str] = (
        "rotor_0_joint",
        "rotor_1_joint",
        "rotor_2_joint",
        "rotor_3_joint",
    )

    mass: float = 1.5
    inertia: tuple[float, float, float] = (0.03, 0.03, 0.06)
    rotor_config: dict = ROTOR_CONFIG

    v_max: float = 5.0   # max linear velocity [m/s]
    w_max: float = 2.0   # max angular velocity [rad/s]

    scale: tuple[float, ...] = (1.0,) * 6
    offset: tuple[float, ...] = (0.0,) * 6
    bounding_strategy: str | None = "clip"

