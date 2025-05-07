import argparse

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(
    description="Collecting dataset for mapping environment."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Parameter for teleoperation.")
parser.add_argument("--decimation", type=int, default=2, help="Parameter for teleoperation.")
parser.add_argument("--dataset_dir", type=str, default="dataset", help="Output directory for the dataset.")
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Reset everything follows."""

import gymnasium as gym
import numpy as np
import swimlab  # noqa: F401
import swimlab_tasks  # noqa: F401
import os
import torch
from scipy.spatial.transform import Rotation as R

import isaaclab.utils.math as math_utils
from isaaclab.devices import Se3Keyboard
from isaaclab_tasks.utils import parse_env_cfg


def pre_process_actions(
    body_pose: torch.Tensor,
    delta_pose: torch.Tensor,
) -> torch.Tensor:
    """
    body_pose: (N, 7) tensor -> (pos_x, pos_y, pos_z, rot_w, rot_x, rot_y, rot_z)
    delta_pose: (N, 6) tensor -> (delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw)
    """
    pos = body_pose[:, :3]  # (pos_x, pos_y, pos_z)
    quat = body_pose[:, 3:] # (rot_w, rot_x, rot_y, rot_z)
    yaw = math_utils.euler_xyz_from_quat(quat)[2]

    # delta
    delta_x = delta_pose[:, 0]
    delta_y = delta_pose[:, 1]
    delta_yaw = delta_pose[:, 5]

    # calculate cos/sin
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    # Update world pose
    delta_x_world = cos_yaw * delta_x - sin_yaw * delta_y
    delta_y_world = sin_yaw * delta_x + cos_yaw * delta_y
    pos[:, 0] += delta_x_world
    pos[:, 1] += delta_y_world

    yaw += delta_yaw

    new_quat = torch.zeros_like(quat)
    new_quat[:, 0] = torch.cos(yaw / 2.0)
    new_quat[:, 1] = 0.0
    new_quat[:, 2] = 0.0
    new_quat[:, 3] = torch.sin(yaw / 2.0)

    new_body_pose = torch.cat([pos, new_quat], dim=-1)

    return new_body_pose


def enough_to_close(
    body_pose: np.array,
    camera_pose: np.array,
    pos_threshold: float = 0.001,
    rot_threshold: float = 0.001,
) -> bool:
    """
    Check if the distance between body pose and camera pose is less than the threshold.

    Args:
        body_pose (np.array): Body pose (3,) in meters.
        camera_pose (np.array): Camera pose (3,) in meters.
        threshold (float): Distance threshold.

    Returns:
        bool: True if the distance is less than the threshold, False otherwise.
    """
    assert body_pose.shape == (7,)
    assert camera_pose.shape == (7,)

    # Calculate the distance between body and camera poses
    pos_distance = np.linalg.norm(body_pose[:3] - camera_pose[:3])
    body_euler = R.from_quat(body_pose[3:]).as_euler("xyz")
    camera_euler = R.from_quat(camera_pose[3:]).as_euler("xyz")
    rot_distance = np.linalg.norm(body_euler - camera_euler)
    return pos_distance < pos_threshold and rot_distance < rot_threshold


def main():
    """Main function."""
    # Create environment configuration
    env_cfg = parse_env_cfg(task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # Disable gravity
    env_cfg.sim.gravity = (0.0, 0.0, 0.0)
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Print info
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # Reset environment
    env.reset()

    # Flags for controlling teleoperation flow
    finish_recording = False
    teleoperation_active = False

    # Callback handlers
    def stop_teleoperation():
        nonlocal finish_recording
        finish_recording = True

    def start_teleoperation():
        nonlocal teleoperation_active
        teleoperation_active = True

    # Create controller
    teleop_interface = Se3Keyboard(
        pos_sensitivity=0.3 * args_cli.sensitivity, rot_sensitivity=0.15 * args_cli.sensitivity
    )
    teleop_interface.add_callback("R", stop_teleoperation)
    teleop_interface.add_callback("F", start_teleoperation)
    teleop_interface.reset()

    # Get robot instance
    robot = env.scene["robot"]
    camera = env.scene["camera"]
    print("[INFO] Camera Intrinsic Parameters:", camera.data.intrinsic_matrices[0].cpu().flatten().tolist())

    rgb_dataset, depth_dataset, pose_dataset = [], [], []

    # Execute simulation
    episode_index = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # Get device command
            delta_pose, _ = teleop_interface.advance()

            # Only apply teleop commands when active
            if teleoperation_active:
                # Compute actions based on teleoperation commands
                body_pos, body_quat = robot.data.root_pos_w, robot.data.root_quat_w
                body_pose = torch.cat((body_pos, body_quat), dim=1)

                # Convert to torch
                delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=body_pose.device).repeat(args_cli.num_envs, 1)

                # Update body pose
                body_pose = pre_process_actions(body_pose, delta_pose)

                # Step simulation
                for _ in range(args_cli.decimation):
                    robot.write_root_pose_to_sim(body_pose)
                    env.sim.render()
                    env.scene.update(env.sim.cfg.dt)

                # Collect data
                rgb_frame = camera.data.output["rgb"].squeeze(0).cpu().numpy()
                depth_frame = camera.data.output["distance_to_image_plane"].squeeze(0).cpu().numpy()
                camera_pos = camera.data.pos_w
                camera_rot = math_utils.convert_quat(camera.data.quat_w_opengl, "xyzw")
                camera_pose = torch.cat((camera_pos, camera_rot), dim=1).squeeze(0).cpu().numpy()

                # Skil if the teleoperation commands is none
                if torch.sum(delta_pose) == 0.0:
                    continue

                # For debug
                # _camera_pos = camera.data.pos_w
                # _camera_rot = camera.data.quat_w_world
                # _camera_pose = torch.cat((_camera_pos, _camera_rot), dim=1).squeeze(0).cpu().numpy()

                # Check if the camera pose is close to the body pose
                # body_pose = body_pose.squeeze(0).cpu().numpy()
                # body_pose[2] = body_pose[2] - 0.1 # Adjust height
                # if not enough_to_close(body_pose, _camera_pose, pos_threshold=0.001, rot_threshold=0.001):
                #     print("Body pose {}: camera pose:{}".format(body_pose, _camera_pose))
                #     continue

                # Append frame and pose
                rgb_dataset.append(rgb_frame)
                depth_dataset.append(depth_frame)
                pose_dataset.append(camera_pose)
            else:
                env.sim.render()

            if finish_recording:
                # Save dataset
                rgb_dataset_dir = os.path.join(args_cli.dataset_dir, "rgb")
                depth_dataset_dir = os.path.join(args_cli.dataset_dir, "depth")
                pose_dataset_dir = os.path.join(args_cli.dataset_dir, "pose")
                os.makedirs(rgb_dataset_dir, exist_ok=True)
                os.makedirs(depth_dataset_dir, exist_ok=True)
                os.makedirs(pose_dataset_dir, exist_ok=True)
                np.save(os.path.join(rgb_dataset_dir, f"rgb_{episode_index}.npy"), rgb_dataset)
                np.save(os.path.join(depth_dataset_dir, f"depth_{episode_index}.npy"), depth_dataset)
                np.save(os.path.join(pose_dataset_dir, f"pose_{episode_index}.npy"), pose_dataset)

                # Reset simulation environment
                env.reset()
                finish_recording = False
                rgb_dataset, depth_dataset, pose_dataset = [], [], []
                episode_index += 1
                user_input = input("Finish recording: Do you want to continue? (y/n): ")
                if user_input.lower() == "y":
                    break
                else:
                    teleoperation_active = False
                    continue
    # Shutdown
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
