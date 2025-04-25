from typing import Sequence
from isaaclab.utils import configclass
from .lee_controller import LeeController, AttitudeController, RateController


@configclass
class LeeControllerCfg:
    class_type: type = LeeController
    controller_type: str = "position"
    gravity: float = 9.81
    position_gain: float | Sequence[float] = (1.0, 1.0, 1.0)
    velocity_gain: float | Sequence[float] = (1.0, 1.0, 1.0)
    attitude_gain: float | Sequence[float] = (1.0, 1.0, 1.0)
    angular_rate_gain: float | Sequence[float] = (1.0, 1.0, 1.0)


@configclass
class AttitudeControllerCfg:
    class_type: type = AttitudeController
    controller_type: str = "position"
    gain_attitude: Sequence[float] = (1.0, 1.0, 1.0)
    gain_angular_rate: Sequence[float] = (1.0, 1.0, 1.0)


@configclass
class RateControllerCfg:
    class_type: type = RateController
    controller_type: str = "position"
    gain_angular_rate: Sequence[float] = (1.0, 1.0, 1.0)

