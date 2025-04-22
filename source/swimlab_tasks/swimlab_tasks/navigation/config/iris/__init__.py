import gymnasium as gym

from . import agents
from . import iris_env_cfg


gym.register(
    id="Isaac-Plain-Navigation-IRIS-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": iris_env_cfg.IRISPlainNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:IRISPlainNavigationPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-LivingRoom-Navigation-IRIS-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": iris_env_cfg.IRISLivingRoomNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:IRISLivingRoomNavigationPPORunnerCfg",
    },
    disable_env_checker=True,
)

