import argparse
import os
import math
from pathlib import Path

import numpy as np
import gdown
import torch
import torchvision.transforms as transforms

import clip

from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from swimlab_navigation.vlmaps.utils.mapping_utils import (
    save_map,
    depth2pc,
    transform_pc,
    get_sim_cam_mat_with_params,
    pos2grid_id,
    project_point,
)
from swimlab_navigation.vlmaps.lseg.modules.models.lseg_net import LSegEncNet
from swimlab_navigation.vlmaps.lseg.additional_utils.models import resize_image, pad_image, crop_image

parser = argparse.ArgumentParser(description="Generate LSeg-based VLMaps.")
parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset directory containing rgb, depth, pose folders.")
parser.add_argument("--camera_height", type=float, default=1.5, help="Height of the camera above ground.")
parser.add_argument("--cs", type=float, default=0.05, help="Cell size (meters) for top-down map grid.")
parser.add_argument("--gs", type=int, default=1000, help="Grid size (number of cells per axis) for top-down map.")
parser.add_argument("--depth_sample_rate", type=int, default=100, help="Subsampling rate for depth points.")
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
    pos_new = np.array([pos[0], -pos[1], pos[2]], dtype=np.float32)

    # Convert quaternion to rotation matrix
    rot_mat = R.from_quat(quat).as_matrix()

    # Flip the y-axis for rotation matrix
    flip_y = np.diag([1, -1, -1])
    rot_mat_new = flip_y @ rot_mat @ flip_y

    # Convert back to quaternion
    quat_new = R.from_matrix(rot_mat_new).as_quat()

    return pos_new, quat_new


def get_lseg_feat(
    model: LSegEncNet,
    image: np.ndarray,
    labels: np.ndarray,
    transform: np.ndarray,
    crop_size: int = 480,
    base_size: int = 520,
    norm_mean: list[float] = [0.5, 0.5, 0.5],
    norm_std: list[float] = [0.5, 0.5, 0.5],
):
    image = transform(image).unsqueeze(0).cuda()
    img = image[0].permute(1,2,0)
    img = img * 0.5 + 0.5
    
    batch, _, h, w = image.size()
    stride_rate = 2.0 / 3.0
    stride = int(crop_size * stride_rate)

    long_size = base_size
    if h > w:
        height = long_size
        width = int(1.0 * w * long_size / h + 0.5)
        short_size = width
    else:
        width = long_size
        height = int(1.0 * h * long_size / w + 0.5)
        short_size = height

    cur_img = resize_image(image, height, width, **{'mode': 'bilinear', 'align_corners': True})

    if long_size <= crop_size:
        pad_img = pad_image(cur_img, norm_mean,
                            norm_std, crop_size)
        print(pad_img.shape)
        with torch.no_grad():
            outputs, logits = model(pad_img, labels)
        outputs = crop_image(outputs, 0, height, 0, width)
    else:
        if short_size < crop_size:
            # pad if needed
            pad_img = pad_image(cur_img, norm_mean, norm_std, crop_size)
        else:
            pad_img = cur_img
        _, _, ph, pw = pad_img.shape
        assert(ph >= height and pw >= width)
        h_grids = int(math.ceil(1.0 * (ph-crop_size)/stride)) + 1
        w_grids = int(math.ceil(1.0 * (pw-crop_size)/stride)) + 1
        with torch.cuda.device_of(image):
            with torch.no_grad():
                outputs = image.new().resize_(batch, model.out_c, ph, pw).zero_().cuda()
                logits_outputs = image.new().resize_(batch, len(labels), ph, pw).zero_().cuda()
            count_norm = image.new().resize_(batch, 1, ph, pw).zero_().cuda()

        # grid evaluation
        for idh in range(h_grids):
            for idw in range(w_grids):
                h0 = idh * stride
                w0 = idw * stride
                h1 = min(h0 + crop_size, ph)
                w1 = min(w0 + crop_size, pw)
                crop_img = crop_image(pad_img, h0, h1, w0, w1)

                # pad if needed
                pad_crop_img = pad_image(crop_img, norm_mean, norm_std, crop_size)
                with torch.no_grad():
                    output, logits = model(pad_crop_img, labels)
                cropped = crop_image(output, 0, h1-h0, 0, w1-w0)
                cropped_logits = crop_image(logits, 0, h1-h0, 0, w1-w0)
                outputs[:, :, h0:h1, w0:w1] += cropped
                logits_outputs[:, :, h0:h1, w0:w1] += cropped_logits
                count_norm[:, :, h0:h1, w0:w1] += 1

        assert((count_norm==0).sum()==0)
        outputs = outputs / count_norm
        logits_outputs = logits_outputs / count_norm
        outputs = outputs[:, :, :height, :width]
        logits_outputs = logits_outputs[:, :, :height, :width]

    outputs = outputs.cpu()
    outputs = outputs.numpy() # B, D, H, W

    return outputs


def create_lseg_map_batch(
    dataset_dir,
    camera_height: float,
    cs: float = 0.05,
    gs: int = 1000,
    depth_sample_rate: int = 100
):
    mask_version = 1 # 0, 1

    crop_size = 480 # 480
    base_size = 520 # 520
    lang = "door,chair,ground,ceiling,other"
    labels = lang.split(",")

    # loading models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_version = "ViT-B/32"
    clip_feat_dim = {
        'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
        'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768
    }[clip_version]
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load(clip_version)
    clip_model.to(device).eval()
    lang_token = clip.tokenize(labels)
    lang_token = lang_token.to(device)

    with torch.no_grad():
        text_feats = clip_model.encode_text(lang_token)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    text_feats = text_feats.cpu().numpy()
    model = LSegEncNet(
        lang,
        arch_option=0,
        block_depth=0,
        activation='lrelu',
        crop_size=crop_size
    )
    model_state_dict = model.state_dict()

    # Downlaod checkpoint
    checkpoint_dir = Path(__file__).resolve().parents[1] / "navigation" / "lseg" / "checkpoints"
    checkpoint_path = checkpoint_dir / "demo_e200.ckpt"
    os.makedirs(checkpoint_dir, exist_ok=True)
    if not checkpoint_path.exists():
        print("Downloading LSeg checkpoint...")
        checkpoint_url = "https://drive.google.com/u/0/uc?id=1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb"
        gdown.download(checkpoint_url, output=str(checkpoint_path))

    pretrained_state_dict = torch.load(checkpoint_path)
    pretrained_state_dict = {k.lstrip('net.'): v for k, v in pretrained_state_dict['state_dict'].items()}
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(pretrained_state_dict)

    model.eval()
    model = model.cuda()

    norm_mean= [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

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
    weight = np.zeros((gs, gs), dtype=float)
    obstacles = np.ones((gs, gs), dtype=np.uint8)

    save_map(color_top_down_save_path, color_top_down)
    save_map(grid_save_path, grid)
    save_map(weight_save_path, weight)
    save_map(obstacles_save_path, obstacles)

    tf_list = []

    rgb_data = np.load(rgb_list[0])  # TODO: Fix this
    depth_data = np.load(depth_list[0])  # TODO: Fix this
    pose_data = np.load(pose_list[0])  # TODO: Fix this
    data_iter = zip(rgb_data, depth_data, pose_data)
    pbar = tqdm(total=len(rgb_data))

    # load all images and depths and poses
    for data_sample in data_iter:
        rgb, depth, pose = data_sample

        pos, rot = pose[:3], pose[3:]
        pos, rot = convert_pose(pos, rot)
        pos[1] += camera_height
        rot = R.from_quat(rot).as_matrix()

        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = pos.reshape(-1)

        tf_list.append(pose)
        if len(tf_list) == 1:
            init_tf_inv = np.linalg.inv(tf_list[0]) 

        tf = init_tf_inv @ pose

        pix_feats = get_lseg_feat(model, rgb, labels, transform, crop_size, base_size, norm_mean, norm_std)

        rgb_cam_mat = get_sim_cam_mat_with_params(
            focal_length=1.66,
            horizontal_aperture=1.89,
            vertical_aperture=1.44,
            width=640,
            height=480,
        )
        feat_cam_mat = get_sim_cam_mat_with_params(
            focal_length=1.66,
            horizontal_aperture=1.89,
            vertical_aperture=1.44,
            width=pix_feats.shape[2],
            height=pix_feats.shape[3]
        )

        # transform all points to the global frame
        pc, mask = depth2pc(depth, rgb_cam_mat)
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
    create_lseg_map_batch(
        args_cli.dataset_dir,
        camera_height=args_cli.camera_height,
        cs=args_cli.cs,
        gs=args_cli.gs,
        depth_sample_rate=args_cli.depth_sample_rate,
    )
