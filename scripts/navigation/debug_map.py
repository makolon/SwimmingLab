import argparse
import os
import math
from pathlib import Path

import numpy as np
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
parser.add_argument("--camera_height", type=float, default=1.5, help="Height of the camera above ground.")
parser.add_argument("--cs", type=float, default=0.05, help="Cell size (meters) for top-down map grid.")
parser.add_argument("--gs", type=int, default=1000, help="Grid size (number of cells per axis) for top-down map.")
parser.add_argument("--depth_sample_rate", type=int, default=100, help="Subsampling rate for depth points.")
args_cli = parser.parse_args()


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
    grid_save_path = os.path.join(map_save_dir, f"grid_lseg_{mask_version}.npy")
    weight_save_path = os.path.join(map_save_dir, f"weight_lseg_{mask_version}.npy")
    obstacles_save_path = os.path.join(map_save_dir, "obstacles.npy")

    # initialize a grid with zero position at the center
    color_top_down_height = (camera_height + 1) * np.ones((gs, gs), dtype=np.float32)
    color_top_down = np.zeros((gs, gs, 3), dtype=np.uint8)
    grid = np.zeros((gs, gs, clip_feat_dim), dtype=np.float32)
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
        feat_cam_mat = get_sim_cam_mat(pix_feats.shape[2], pix_feats.shape[3])  # TODO: Fix this

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

            # average the visual embeddings if multiple points are projected to the same grid cell
            px, py, pz = project_point(feat_cam_mat, p_local)
            if not (px < 0 or py < 0 or px >= pix_feats.shape[3] or py >= pix_feats.shape[2]):
                feat = pix_feats[0, :, py, px]
                grid[y, x] = (grid[y, x] * weight[y, x] + feat) / (weight[y, x] + 1)
                weight[y, x] += 1

            # build an obstacle map ignoring points on the floor (0 means occupied, 1 means free)
            if p_local[1] > camera_height:
                continue
            obstacles[y, x] = 0
        pbar.update(1)

    save_map(color_top_down_save_path, color_top_down)
    save_map(grid_save_path, grid)
    save_map(weight_save_path, weight)
    save_map(obstacles_save_path, obstacles)


if __name__ == "__main__":
    debug_map(
        args_cli.dataset_dir,
        camera_height=args_cli.camera_height,
        cs=args_cli.cs,
        gs=args_cli.gs,
        depth_sample_rate=args_cli.depth_sample_rate,
    )
