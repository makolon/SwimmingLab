import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import torch
import clip

from swimlab_navigation.vlmaps.utils.mapping_utils import (
    load_map,
    get_new_pallete,
    get_new_mask_pallete,
)
from swimlab_navigation.vlmaps.utils.clip_utils import get_text_feats


def visualize_maps(
    obstacles_save_path: Path,
    color_top_down_save_path: Path,
    grid_save_path: Path,
):
    obstacles = load_map(obstacles_save_path)
    x_indices, y_indices = np.where(obstacles == 0)
    xmin = np.min(x_indices)
    xmax = np.max(x_indices)
    ymin = np.min(y_indices)
    ymax = np.max(y_indices)

    print(np.unique(obstacles))
    obstacles_pil = Image.fromarray(obstacles[xmin : xmax + 1, ymin : ymax + 1])
    plt.figure(figsize=(8, 6), dpi=120)
    plt.imshow(obstacles_pil, cmap="gray")
    plt.savefig("obstacles.png")

    color_top_down = load_map(color_top_down_save_path)
    color_top_down = color_top_down[xmin : xmax + 1, ymin : ymax + 1]
    color_top_down_pil = Image.fromarray(color_top_down)
    plt.figure(figsize=(8, 6), dpi=120)
    plt.imshow(color_top_down_pil)
    plt.savefig("color_top_down.png")

    # Prepare clip model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_version = "ViT-B/32"
    clip_feat_dim = {
        "RN50": 1024,
        "RN101": 512,
        "RN50x4": 640,
        "RN50x16": 768,
        "RN50x64": 1024,
        "ViT-B/32": 512,
        "ViT-B/16": 512,
        "ViT-L/14": 768,
    }[clip_version]
    clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
    clip_model.to(device).eval()

    grid = load_map(grid_save_path)
    grid = grid[xmin : xmax + 1, ymin : ymax + 1]

    no_map_mask = obstacles[xmin : xmax + 1, ymin : ymax + 1] > 0
    print(no_map_mask.shape)

    lang = "door,chair,ground,ceiling,other"  # TODO: Fix this
    lang = lang.split(",")
    text_feats = get_text_feats(lang, clip_model, clip_feat_dim)

    map_feats = grid.reshape((-1, grid.shape[-1]))
    scores_list = map_feats @ text_feats.T

    predicts = np.argmax(scores_list, axis=1)
    predicts = predicts.reshape((xmax - xmin + 1, ymax - ymin + 1))
    floor_mask = predicts == 2

    new_pallete = get_new_pallete(len(lang))
    mask, patches = get_new_mask_pallete(
        predicts, new_pallete, out_label_flag=True, labels=lang
    )
    seg = mask.convert("RGBA")
    seg = np.array(seg)
    seg[no_map_mask] = [225, 225, 225, 255]
    seg[floor_mask] = [225, 225, 225, 255]
    seg = Image.fromarray(seg)
    plt.figure(figsize=(10, 6), dpi=120)
    plt.legend(
        handles=patches, loc="upper left", bbox_to_anchor=(1.0, 1), prop={"size": 10}
    )
    plt.axis("off")
    plt.title("VLMaps")
    plt.imshow(seg)
    plt.savefig("vlmap.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize VLMaps or obstacle maps from saved grids."
    )
    parser.add_argument(
        "--map_dir", type=str, required=True, help="Path to the saved map directory."
    )
    args = parser.parse_args()

    map_dir = Path(args.map_dir)
    obstacles_save_path = map_dir / "obstacles.npy"
    color_top_down_save_path = map_dir / "color_top_down_1.npy"
    grid_save_path = map_dir / "grid_lseg_1.npy"

    visualize_maps(obstacles_save_path, color_top_down_save_path, grid_save_path)
