from dataclasses import MISSING
from typing import Sequence, Tuple

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from swimlab.envs.mdp.actions import rotor_actions


@configclass
class RotorActionCfg(ActionTermCfg):
    """Config for position, velocity, acceleration + yaw rate action mapped."""
    class_type: type[ActionTerm] = rotor_actions.RotorAction

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
    rotor_params: dict[str, float] | None = None
    """Parameters for the thrust and body force / torque calculation. Defauls to None, in which case the default parameters for the method are use."""

    def __post_init__(self):
        default_rotor_params = {
            "num_rotors": 4,
            "rotor_angles": [0.0, 0.0, 0.0, 0.0],
            "arm_lengths": [0.0, 0.0, 0.0, 0.0],
            "force_constants": [0.0] * 4,
            "moment_constants": [0.0] * 4,
            "kf": [0.0] * 4,
            "km": [0.0] * 4,
            "directions": [1, 1, 1, 1],
            "max_rotation_velocities": [0.0] * 4,
            "mass": 0.0,
            "inertia": {"xx": 0.0, "xy": 0.0, "xz": 0.0, "yy": 0.0, "yz": 0.0, "zz": 0.0},
            "drag_coef": 0.0,
            "tau_up": [0.0] * 4,
            "tau_down": [0.0] * 4,
        }
        rotor_params = default_rotor_params.copy()
        if self.rotor_params is not None:
            rotor_params.update(self.rotor_params)
        self.rotor_params = rotor_params

