from typing import Sequence
from isaaclab.utils import configclass
from .lee_position_controller import LeePositionController, AttitudeController, RateController


@configclass
class LeePositionControllerCfg:
    class_type: type = LeePositionController
    gravity: float = 9.81
    position_gain: float | Sequence[float] = (1.0, 1.0, 1.0)
    velocity_gain: float | Sequence[float] = (1.0, 1.0, 1.0)
    attitude_gain: float | Sequence[float] = (1.0, 1.0, 1.0)
    angular_rate_gain: float | Sequence[float] = (1.0, 1.0, 1.0)


@configclass
class AttitudeControllerCfg:
    class_type: type = AttitudeController
    gain_attitude: Sequence[float] = (1.0, 1.0, 1.0)
    gain_angular_rate: Sequence[float] = (1.0, 1.0, 1.0)


@configclass
class RateControllerCfg:
    class_type: type = RateController
    gain_angular_rate: Sequence[float] = (1.0, 1.0, 1.0)

