import gymnasium as gym

from . import agents
from . import rotor_env_cfg
from . import lee_vel_env_cfg


##
# Lee Velocity
##

gym.register(
    id="Isaac-Plane-ObstacleNavigation-HummingBird-LeeVel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": lee_vel_env_cfg.HummingBirdPlaneObstacleNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HummingBirdPlaneObstacleNavigationPPORunnerCfg",
    },
    disable_env_checker=True,
)

##
# Rotor Action
##

gym.register(
    id="Isaac-Plane-ObstacleNavigation-HummingBird-Rotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": rotor_env_cfg.HummingBirdPlaneObstacleNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HummingBirdPlaneObstacleNavigationPPORunnerCfg",
    },
    disable_env_checker=True,
)

