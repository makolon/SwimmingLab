from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass


##
# Pre-defined configuration
##
from .base_scene_cfg import BaseSceneCfg

##
# Scene definition
##
@configclass
class MatterportSceneCfg(BaseSceneCfg):
    """Configuration for the living room scene with a robot and multiple objects.
    """


