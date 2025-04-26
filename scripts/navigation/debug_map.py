import argparse
import os
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2
import gdown
import torch

import clip

from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from swimlab_navigation.vlmaps.utils.mapping_utils import (
    load_pose,
    save_map,
    depth2pc,
    transform_pc,
    get_sim_cam_mat,
    pos2grid_id,
    project_point,
)
from swimlab_navigation.vlmaps.lseg.additional_utils.models import resize_image, pad_image, crop_image

parser = argparse.ArgumentParser(description="Debug map consturction.")
parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset directory containing rgb, depth, pose folders.")
parser.add_argument("--camera_height", type=float, default=1.4, help="Height of the camera above ground.")
parser.add_argument("--cs", type=float, default=0.05, help="Cell size (meters) for top-down map grid.")
parser.add_argument("--gs", type=int, default=1000, help="Grid size (number of cells per axis) for top-down map.")
parser.add_argument("--depth_sample_rate", type=int, default=100, help="Subsampling rate for depth points.")
args_cli = parser.parse_args()


def transform_quat(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion from (x forward, y right, z up) frame
    to (x right, y down, z forward) frame.

    Args:
        quat (np.ndarray): (4) quaternion in (x, y, z, w) format

    Returns:
        np.ndarray: (4) transformed quaternion in (x, y, z, w) format
    """
    assert quat.shape[-1] == 4, "Input should be (4) for quaternion (xyzw format)."

    # Convert quaternion to rotation matrix
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]

    # Rotation matrix (batch)
    rot = np.zeros((3, 3))

    rot[0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    rot[0, 1] = 2 * (x * y - z * w)
    rot[0, 2] = 2 * (x * z + y * w)
    rot[1, 0] = 2 * (x * y + z * w)
    rot[1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    rot[1, 2] = 2 * (y * z - x * w)
    rot[2, 0] = 2 * (x * z - y * w)
    rot[2, 1] = 2 * (y * z + x * w)
    rot[2, 2] = 1 - 2 * (x ** 2 + y ** 2)

    # Coordinate axis transformation matrix
    # Old basis: (x, y, z)
    # New basis: (y, -z, x)
    transform = np.array([
        [0, 0, 1],
        [0, -1, 0],
        [1, 0, 0]
    ], dtype=np.float32)  # (3,3)

    # Apply basis change: R_new = T * R_old * T^T
    rot_new = transform @ rot @ transform.T

    # Convert rotation matrix back to quaternion
    qw = 0.5 * np.sqrt(1 + rot_new[0, 0] + rot_new[1, 1] + rot_new[2, 2])
    qx = (rot_new[2, 1] - rot_new[1, 2]) / (4 * qw)
    qy = (rot_new[0, 2] - rot_new[2, 0]) / (4 * qw)
    qz = (rot_new[1, 0] - rot_new[0, 1]) / (4 * qw)

    quat_new = np.stack([qx, qy, qz, qw])

    return quat_new


def debug_map(
    dataset_dir,
    camera_height: float,
    cs: float = 0.05,
    gs: int = 1000,
    depth_sample_rate: int = 100
):
    mask_version = 1 # 0, 1

    crop_size = 480 # 480
    base_size = 520 # 520

    norm_mean= [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    padding = [0.0] * 3

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
    color_top_down_save_path = os.path.join(map_save_dir, f"color_top_down_{mask_version}.npy")
    weight_save_path = os.path.join(map_save_dir, f"weight_lseg_{mask_version}.npy")
    obstacles_save_path = os.path.join(map_save_dir, "obstacles.npy")

    # initialize a grid with zero position at the center
    color_top_down_height = (camera_height + 1) * np.ones((gs, gs), dtype=np.float32)
    color_top_down = np.zeros((gs, gs, 3), dtype=np.uint8)
    obstacles = np.ones((gs, gs), dtype=np.uint8)
    weight = np.zeros((gs, gs), dtype=float)

    tf_list = []

    rgb_data = np.load(rgb_list[0])  # TODO: Fix this
    depth_data = np.load(depth_list[0])  # TODO: Fix this
    pose_data = np.load(pose_list[0])  # TODO: Fix this
    data_iter = zip(rgb_data, depth_data, pose_data)
    pbar = tqdm(total=len(rgb_data))

    # load all images and depths and poses
    for data_sample in data_iter:
        rgb, depth, pose = data_sample

        # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pos, rot = pose[:3], pose[3:]
        rot = transform_quat(rot)
        rot_mat = R.from_quat(rot).as_matrix()
        rot_ro_cam = np.eye(3)
        rot_ro_cam[1, 1] = -1
        rot_ro_cam[2, 2] = -1
        rot = rot_mat @ rot_ro_cam
        pos[1] += camera_height

        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = pos.reshape(-1)

        tf_list.append(pose)
        if len(tf_list) == 1:
            init_tf_inv = np.linalg.inv(tf_list[0]) 

        tf = init_tf_inv @ pose

        # transform all points to the global frame
        pc, mask = depth2pc(depth)
        shuffle_mask = np.arange(pc.shape[1]) 
        np.random.shuffle(shuffle_mask)
        shuffle_mask = shuffle_mask[::depth_sample_rate]
        mask = mask[shuffle_mask]
        pc = pc[:, shuffle_mask]
        pc = pc[:, mask]
        pc_global = transform_pc(pc, tf)

        rgb_cam_mat = get_sim_cam_mat(rgb.shape[0], rgb.shape[1])  # TODO: Fix this

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
        axes[0].set_title(f"Color Top-Down Map")
        axes[0].axis("off")
        # obstacles
        axes[1].imshow(obstacles, cmap="gray")
        axes[1].set_title(f"Obstacle Map")
        axes[1].axis("off")
        plt.tight_layout()

        save_dir = os.path.join(map_save_dir, "viz")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"map_step.png"))


if __name__ == "__main__":
    debug_map(
        args_cli.dataset_dir,
        camera_height=args_cli.camera_height,
        cs=args_cli.cs,
        gs=args_cli.gs,
        depth_sample_rate=args_cli.depth_sample_rate,
    )
