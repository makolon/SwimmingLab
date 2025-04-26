import gymnasium as gym

from . import agents
from . import rotor_env_cfg
from . import lee_vel_env_cfg


##
# Lee Velocity
##

gym.register(
    id="Isaac-Plane-ObstacleNavigation-IRIS-LeeVel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": lee_vel_env_cfg.IRISPlaneObstacleNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:IRISPlaneObstacleNavigationPPORunnerCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-LivingRoom-ObstacleNavigation-IRIS-LeeVel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": lee_vel_env_cfg.IRISLivingRoomObstacleNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:IRISLivingRoomObstacleNavigationPPORunnerCfg",
    },
    disable_env_checker=True,
)

##
# Rotor Action
##

gym.register(
    id="Isaac-Plane-ObstacleNavigation-IRIS-Rotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": rotor_env_cfg.IRISPlaneObstacleNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:IRISPlaneObstacleNavigationPPORunnerCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-LivingRoom-ObstacleNavigation-IRIS-Rotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": rotor_env_cfg.IRISLivingRoomObstacleNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:IRISLivingRoomObstacleNavigationPPORunnerCfg",
    },
    disable_env_checker=True,
)

