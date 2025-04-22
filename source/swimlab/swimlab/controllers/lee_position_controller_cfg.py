from dataclasses import MISSING
from typing import Literal, Sequence
from isaaclab.utils import configclass
from .lee_position_controller import LeePositionController, AttitudeController, RateController


@configclass
class LeePositionControllerCfg:
    """Configuration for SE3 Lee position controller."""
    class_type: type = LeePositionController

    mass: float = 1.0
    """The mass of the rigid body."""
    gravity: float = 9.81
    """The gravity of the simulation."""

    position_gain: float | Sequence[float] = (1.0, 1.0, 1.0)
    """The position gain for the p-gain controller."""
    velocity_gain: float | Sequence[float] = (1.0, 1.0, 1.0)
    """The velocity gain for the d-gain controller."""
    attitude_gain: float | Sequence[float] = (1.0, 1.0, 1.0)
    """The attitude gain for the attitute controller."""
    angular_rate_gain: float | Sequence[float] = (1.0, 1.0, 1.0)
    """The angular rate gain for the rate controller."""

    rotor_params: dict[str, float] | None = None
    """Parameters for the lee controller. Defaults to None, in which case the default
    parameters for the method are used."""

    def __post_init__(self):
        # default parameters of the rotor.
        default_rotor_params = {
            "rotor_angles": [0.0, 0.0, 0.0, 0.0],
            "arm_lengths": [0.0, 0.0, 0.0, 0.0],
            "force_constants": [0.0, 0.0, 0.0, 0.0],
            "moment_constants": [0.0, 0.0, 0.0, 0.0],
            "directions": [1.0, 1.0, 1.0, 1.0],
            "max_rotation_velocities": [1.0, 1.0, 1.0, 1.0],
            "inertia": {"xx": 1.0, "xy": 0.0, "xz": 0.0, "yy": 1.0, "yz": 0.0, "zz": 1.0},
        }
        # update parameters for IK-method if not provided
        rotor_params = default_rotor_params.copy()
        if self.rotor_params is not None:
            rotor_params.update(self.rotor_params)
        self.rotor_params = rotor_params


@configclass
class AttitudeControllerCfg:
    """Configuration for attitude controller."""
    class_type: type = AttitudeController

    mass: float = 1.0
    """The mass of the rigid body."""
    gravity: float = 9.81
    """The gravity of the simulation."""

    gain_attitude: Sequence[float] = (1.0, 1.0, 1.0)
    """The gain attitude for the attitude controller."""
    gain_angular_rate: Sequence[float] = (1.0, 1.0, 1.0)
    """The angular rate gain for the attitude controller."""

    rotor_params: dict[str, float] | None = None
    """Parameters for the lee controller. Defaults to None, in which case the default
    parameters for the method are used."""

    def __post_init__(self):
        # default parameters of the rotor.
        default_rotor_params = {
            "rotor_angles": [0.0, 0.0, 0.0, 0.0],
            "arm_lengths": [0.0, 0.0, 0.0, 0.0],
            "force_constants": [0.0, 0.0, 0.0, 0.0],
            "moment_constants": [0.0, 0.0, 0.0, 0.0],
            "directions": [1.0, 1.0, 1.0, 1.0],
            "max_rotation_velocities": [1.0, 1.0, 1.0, 1.0],
            "inertia": {"xx": 1.0, "xy": 0.0, "xz": 0.0, "yy": 1.0, "yz": 0.0, "zz": 1.0},
        }
        # update parameters for IK-method if not provided
        rotor_params = default_rotor_params.copy()
        if self.rotor_params is not None:
            rotor_params.update(self.rotor_params)
        self.rotor_params = rotor_params


@configclass
class RateControllerCfg:
    """Configuration for rate controller."""
    class_type: type = RateController

    mass: float = 1.0
    """The mass of the rigid body."""
    gravity: float = 9.81
    """The gravity of the simulation."""

    gain_angular_rate: Sequence[float] = (1.0, 1.0, 1.0)
    """The angular rate gain for the rate controller."""

    rotor_params: dict[str, float] | None = None
    """Parameters for the lee controller. Defaults to None, in which case the default
    parameters for the method are used."""

    def __post_init__(self):
        # default parameters of the rotor.
        default_rotor_params = {
            "rotor_angles": [0.0, 0.0, 0.0, 0.0],
            "arm_lengths": [0.0, 0.0, 0.0, 0.0],
            "force_constants": [0.0, 0.0, 0.0, 0.0],
            "moment_constants": [0.0, 0.0, 0.0, 0.0],
            "directions": [1.0, 1.0, 1.0, 1.0],
            "max_rotation_velocities": [1.0, 1.0, 1.0, 1.0],
            "inertia": {"xx": 1.0, "xy": 0.0, "xz": 0.0, "yy": 1.0, "yz": 0.0, "zz": 1.0},
        }
         # update parameters for IK-method if not provided
        rotor_params = default_rotor_params.copy()
        if self.rotor_params is not None:
            rotor_params.update(self.rotor_params)
        self.rotor_params = rotor_params

