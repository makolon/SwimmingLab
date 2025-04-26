import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# Add argparse arguments
parser = argparse.ArgumentParser(
    description="Collecting dataset for mapping environment."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--root_dir", type=str, default="zarr_dataset", help="Output directory for the dataset.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Reset everything follows."""

import gymnasium as gym
import swimlab  # noqa: F401
import swimlab_tasks  # noqa: F401
import time
import torch
from isaaclab_tasks.utils import parse_env_cfg


def save_to_zarr(data, output_dir, episode_index):
    """Save collected data to a Zarr dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zarr_file = zarr.open(str(output_dir / f"episode_{episode_index:06d}.zarr"), mode="w")

    for key, value in data.items():
        zarr_file.create_dataset(key, data=value, overwrite=True)

    print(f"[INFO]: Zarr dataset saved at {output_dir / f'episode_{episode_index:06d}.zarr'}")


def pre_process_actions(
    policy: Callable, obs: torch.Tensor, teleop_data: tuple[np.adarray, bool]
) -> torch.Tensor:
    commands, _ = teleop_data
    # Convert to torch
    commands = torch.tensor(commands, dtype=torch.float, device=device).repeat(num_envs, 1)
    velocity_commands = torch.cat((
        commands[:, 0],  # move x-axis
        commands[:, 1],  # move y-axis
        commands[:, 2],  # move z-axis
        commands[:, 5],  # rotate z-axis
    ), device=device)
    # Substitute velocity_commands
    obs[:, 9:13] = velocity_commands

    # Agent stepping
    actions = policy(obs)
    return actions


def main():
    """Main function."""
    # Create environment configuration
    env_cfg = parse_env_cfg(task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Wrap the environment
    from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
    env = RslRlVecEnvWrapper(env)

    # Print info
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Reset environment
    obs, _ = env.get_observations()

    # Flag for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True

    # Callback handlers
    def reset_recording_instance():
        """Reset the environment to its initial state.

        This callback is triggered when the user presses the reset key (typically 'R').
        It's useful when:
        - The robot gets into an undesirable configuration
        - The user wants to start over with the task
        - Objects in the scene need to be reset to their initial positions

        The environment will be reset on the next simulation step.
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    # Create controller
    teleop_interface = Se3Keyboard(
        pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity
    )
    teleop_interface.add_callback("R", reset_recording_instance)
    teleop_interface.reset()

    episode_data = []
    episode_index = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # Get device command
            teleop_data = teleop_interface.advance()

            # Only apply teleop commands when active
            if teleoperation_active:
                # Compute actions based on teleoperation commands
                actions = pre_process_actions(policy, obs, teleope_data)
                # Apply actions
                obs, rew, done, extras = env.step(actions)
            else:
                env.sim.render()

            # Collect data
            actions = actions.squeeze(0).cpu().numpy()
            camera_pose = obs["camera_image"]["camera_transform"].squeeze(0).cpu().numpy()
            rgb_frame = obs["camera_image"]["rgb_image"].squeeze(0).cpu().numpy()
            depth_frame = obs["camera_image"]["depth_image"].squeeze(0).cpu().numpy()

            # Append frame and metadata
            episode_data.append({
                "observation.image.rgb": rgb_frame,
                "observation.image.depth": depth_frame,
                "observation.image.pose": camera_pose,
                "action": actions,
                "episode_index": episode_index,
                "frame_index": step,
                "next.reward": rew,
                "next.done": done,
            })

            # Increment step
            step += 1

            if should_reset_recording_instance:
                # Save dataset
                save_to_zarr(
                    data={
                        key: np.array([frame[key] for frame in episode_data])
                        for key in episode-data[0]
                    },
                    output_dir=args_cli.root_dir,
                    episode_index=episode_index,
                )

                # Reset environment
                env.reset()
                should_reset_recording_instance = False

    # Shutdown
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
