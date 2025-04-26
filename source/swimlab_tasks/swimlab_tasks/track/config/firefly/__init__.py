import gymnasium as gym

from . import agents
from . import rotor_env_cfg
from . import lee_vel_env_cfg


##
# Lee Velocity
##

gym.register(
    id="Isaac-Plane-Track-FireFly-LeeVel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": lee_vel_env_cfg.FireFlyPlaneTrackEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FireFlyPlaneTrackPPORunnerCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-LivingRoom-Track-FireFly-LeeVel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": lee_vel_env_cfg.FireFlyLivingRoomTrackEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FireFlyLivingRoomTrackPPORunnerCfg",
    },
    disable_env_checker=True,
)

##
# Rotor Action
##

gym.register(
    id="Isaac-Plane-Track-FireFly-Rotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": rotor_env_cfg.FireFlyPlaneTrackEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FireFlyPlaneTrackPPORunnerCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-LivingRoom-Track-FireFly-Rotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": rotor_env_cfg.FireFlyLivingRoomTrackEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FireFlyLivingRoomTrackPPORunnerCfg",
    },
    disable_env_checker=True,
)


