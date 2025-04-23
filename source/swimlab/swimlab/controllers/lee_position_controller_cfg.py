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

