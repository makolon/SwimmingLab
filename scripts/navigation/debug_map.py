import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from swimlab_navigation.vlmaps.utils.mapping_utils import (
    depth2pc,
    transform_pc,
    get_sim_cam_mat_with_params,
    pos2grid_id,
    project_point,
)

parser = argparse.ArgumentParser(description="Debug map consturction.")
parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset directory containing rgb, depth, pose folders.")
parser.add_argument("--camera_height", type=float, default=2.0, help="Height of the camera above ground.")
parser.add_argument("--cs", type=float, default=0.05, help="Cell size (meters) for top-down map grid.")
parser.add_argument("--gs", type=int, default=1000, help="Grid size (number of cells per axis) for top-down map.")
parser.add_argument("--depth_sample_rate", type=int, default=10, help="Subsampling rate for depth points.")
args_cli = parser.parse_args()


def convert_pose(pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert pose from (x forward, y left, z up) to (x right, y down, z forward).

    Args:
        pos (np.ndarray): Position (3,) in meters.
        quat (np.ndarray): Quaternion (4,) in (x, y, z, w) format.

    Returns:
        tuple[np.ndarray, np.ndarray]: Transformed (position, quaternion).
    """
    assert pos.shape == (3,)
    assert quat.shape == (4,)

    # Flip the y-axis for position
    pos_new = np.array([pos[0], pos[2], -pos[1]], dtype=np.float32)

    # Convert quaternion to rotation matrix
    rot_mat = R.from_quat(quat).as_matrix()

    # Flip the y-axis for rotation matrix
    flip_y = np.diag([1, 1, -1])
    rot_mat_new = flip_y @ rot_mat @ flip_y

    # Convert back to quaternion
    quat_new = R.from_matrix(rot_mat_new).as_quat()

    return pos_new, quat_new


def debug_map(
    dataset_dir,
    camera_height: float,
    cs: float = 0.05,
    gs: int = 1000,
    depth_sample_rate: int = 100
):
    print(f"loading scene {dataset_dir}")
    rgb_dir = os.path.join(dataset_dir, "rgb")
    depth_dir = os.path.join(dataset_dir, "depth")
    pose_dir = os.path.join(dataset_dir, "pose")

    rgb_list = sorted(os.listdir(rgb_dir), key=lambda x: int(
        x.split("_")[-1].split(".")[0]))
    depth_list = sorted(os.listdir(depth_dir), key=lambda x: int(
        x.split("_")[-1].split(".")[0]))
    pose_list = sorted(os.listdir(pose_dir), key=lambda x: int(
        x.split("_")[-1].split(".")[0]))

    rgb_list = [os.path.join(rgb_dir, x) for x in rgb_list]
    depth_list = [os.path.join(depth_dir, x) for x in depth_list]
    pose_list = [os.path.join(pose_dir, x) for x in pose_list]

    map_save_dir = os.path.join(dataset_dir, "map")
    os.makedirs(map_save_dir, exist_ok=True)

    # initialize a grid with zero position at the center
    color_top_down_height = (camera_height + 1) * np.ones((gs, gs), dtype=np.float32)
    color_top_down = np.zeros((gs, gs, 3), dtype=np.uint8)
    obstacles = np.ones((gs, gs), dtype=np.uint8)

    tf_list = []

    rgb_data = np.load(rgb_list[0])  # TODO: Fix this
    depth_data = np.load(depth_list[0])  # TODO: Fix this
    pose_data = np.load(pose_list[0])  # TODO: Fix this
    data_iter = zip(rgb_data, depth_data, pose_data)
    pbar = tqdm(total=len(rgb_data))

    # load all images and depths and poses
    for rgb, depth, pose in data_iter:
        pos, rot = pose[:3], pose[3:]
        pos, rot = convert_pose(pos, rot)
        rot = R.from_quat(rot).as_matrix()

        pose = np.eye(4) 
        pose[:3, :3] = rot
        pose[:3, 3] = pos.reshape(-1)

        tf_list.append(pose)
        if len(tf_list) == 1:
            init_tf_inv = np.linalg.inv(tf_list[0]) 

        tf = init_tf_inv @ pose

        rgb_cam_mat = get_sim_cam_mat_with_params(
            focal_length=1.9299999475479126,
            horizontal_aperture=3.8959999084472656,
            vertical_aperture=2.453000068664551,
            width=1936,
            height=1216,
        )

        # transform all points to the global frame
        pc, mask = depth2pc(depth, intr_mat=rgb_cam_mat)
        shuffle_mask = np.arange(pc.shape[1]) 
        np.random.shuffle(shuffle_mask)
        shuffle_mask = shuffle_mask[::depth_sample_rate]
        mask = mask[shuffle_mask]
        pc = pc[:, shuffle_mask]
        pc = pc[:, mask]
        pc_global = transform_pc(pc, tf)

        # project all point cloud onto the ground
        for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
            x, y = pos2grid_id(gs, cs, p[0], p[2])

            # ignore points projected to outside of the map and points that are 0.5 higher than the camera (could be from the ceiling)
            if x >= obstacles.shape[0] or y >= obstacles.shape[1] or \
                x < 0 or y < 0 or p_local[1] < -0.5:
                continue

            rgb_px, rgb_py, rgb_pz = project_point(rgb_cam_mat, p_local)
            rgb_v = rgb[rgb_py, rgb_px, :]

            # when the projected location is already assigned a color value before, overwrite if the current point has larger height
            if p_local[1] < color_top_down_height[y, x]:
                color_top_down[y, x] = rgb_v
                color_top_down_height[y, x] = p_local[1]

            # build an obstacle map ignoring points on the floor (0 means occupied, 1 means free)
            if p_local[1] > camera_height:
                continue
            obstacles[y, x] = 0

        pbar.update(1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # color_top_down
        axes[0].imshow(color_top_down)
        axes[0].axis("off")
        # obstacles
        axes[1].imshow(obstacles, cmap="gray")
        axes[1].axis("off")
        plt.tight_layout()

        save_dir = os.path.join(map_save_dir, "viz")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "map_step.png"))


if __name__ == "__main__":
    debug_map(
        args_cli.dataset_dir,
        camera_height=args_cli.camera_height,
        cs=args_cli.cs,
        gs=args_cli.gs,
        depth_sample_rate=args_cli.depth_sample_rate,
    )