import argparse

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(
    description="Sample code for demonstrating SwimLabTasks."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
parser.add_argument("--renderer", type=str, default="RaytracedLighting", help="Renderer to use.")
parser.add_argument("--video", action="store_true", help="Record video.")
parser.add_argument(
    "--video_folder", type=str, default="videos", help="Folder to store videos."
)
parser.add_argument(
    "--video_interval", type=int, default=100, help="Interval to record video."
)
parser.add_argument(
    "--video_length", type=int, default=300, help="Length of the video."
)
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import swimlab  # noqa: F401
import swimlab_tasks  # noqa: F401
import time
import torch
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Main function."""
    # Create environment configuration
    env_cfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    if args_cli.headless:
        if args_cli.renderer == "PathTracing":
            # Set the renderer to PathTracing
            viewport_api = get_active_viewport()
            viewport_api.set_hd_engine("rtx", "PathTracing")
        elif args_cli.renderer == "RaytracedLighting":
            # Set the renderer to PathTracing
            viewport_api = get_active_viewport()
            viewport_api.set_hd_engine("rtx", "RaytracedLighting")

    if args_cli.video:
        # Record video
        video_kwargs = {
            "video_folder": args_cli.video_folder,
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Reset environment
    env.reset()

    # Simulate physics
    count = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            # Reset
            if count % 500 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")

            # Sample random actions
            actions = torch.zeros_like(env.action_manager.action)
            actions[:, 0] = 0.1

            # Step the environment
            obs, rew, terminated, truncated, info = env.step(actions)

            # Update counter
            count += 1
        print(f"Time taken for step: {time.time() - start_time}")

    # Close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # Close sim app
    simulation_app.close()
