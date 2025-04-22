from dataclasses import MISSING
from typing import Sequence, Tuple

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from swimlab.controllers import LeePositionControllerCfg
from swimlab.envs.mdp.actions import iris_actions


@configclass
class IRISVelocityActionCfg(ActionTermCfg):
    """Config for velocity + yaw rate action mapped through LeePositionController."""
    class_type: type[ActionTerm] = iris_actions.IRISVelocityAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    body_name: str = MISSING
    """Name of the body or frame for which IK is performed."""
    linear_scale: Sequence[float] | float = (1.0, 1.0, 1.0)
    """Scale factor for the linear velocity scale. Defaults to (1.0, 1.0, 1.0)."""
    yaw_scale: float = 1.0
    """Scale factor for the yaw velocity targets. Defaults to 1.0."""
    controller: LeePositionControllerCfg = MISSING
    """The configuration for the lee position controller."""
