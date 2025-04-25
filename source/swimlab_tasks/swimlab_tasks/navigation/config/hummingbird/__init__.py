import gymnasium as gym

from . import agents
from . import lee_vel_env_cfg


gym.register(
    id="Isaac-Plane-Navigation-HummingBird-LeeVel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": lee_vel_env_cfg.HummingBirdPlaneNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HummingBirdPlaneNavigationPPORunnerCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-LivingRoom-Navigation-HummingBird-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": lee_vel_env_cfg.HummingBirdLivingRoomNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HummingBirdLivingRoomNavigationPPORunnerCfg",
    },
    disable_env_checker=True,
)

