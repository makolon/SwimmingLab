from dataclasses import MISSING
from typing import Sequence, Tuple

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from swimlab.controllers import LeePositionControllerCfg
from swimlab.envs.mdp.actions import drone_actions


@configclass
class DroneVelocityActionCfg(ActionTermCfg):
    """Config for velocity + yaw rate action mapped through LeePositionController."""
    class_type: type[ActionTerm] = drone_actions.DroneVelocityAction

    rotor_joint_names: list[str] = MISSING
    """List of rotor joint names or regex expressions that the action will be mapped to."""
    rotor_body_names: list[str] = MISSING
    """List of rotor body names or regrex expressions that the action will be mapped to."""
    base_body_names: list[str] = MISSING
    """LIst of base body names or regrex expressions that the action will be mapped to."""
    linear_scale: Sequence[float] | float = (1.0, 1.0, 1.0)
    """Scale factor for the linear velocity scale. Defaults to (1.0, 1.0, 1.0)."""
    yaw_scale: float = 1.0
    """Scale factor for the yaw velocity targets. Defaults to 1.0."""
    controller: LeePositionControllerCfg = MISSING
    """The configuration for the lee position controller."""
    rotor_params: dict[str, float] | None = None
    """Parameters for the thrust and body force / torque calculation. Defauls to None, in which case the default parameters for the method are use."""

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
            "kf": 0.0,
            "km": 0.0,
            "drag_coef": 0.0,
            "tau_up": 0.0,
            "tau_down": 0.0,
        }
        # update parameters if not provided
        rotor_params = default_rotor_params.copy()
        if self.rotor_params is not None:
            rotor_params.update(self.rotor_params)
        self.rotor_params = rotor_params
