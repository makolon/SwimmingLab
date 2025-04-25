import gymnasium as gym

from . import agents
from . import lee_vel_env_cfg


gym.register(
    id="Isaac-Plane-Navigation-IRIS-LeeVel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": lee_vel_env_cfg.IRISPlaneNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:IRISPlaneNavigationPPORunnerCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-LivingRoom-Navigation-IRIS-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": lee_vel_env_cfg.IRISLivingRoomNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:IRISLivingRoomNavigationPPORunnerCfg",
    },
    disable_env_checker=True,
)

