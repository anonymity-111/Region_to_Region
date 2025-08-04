import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from typing import Dict
import random
from random import choice, uniform
import ldm.models.pietorch as pietorch


def get_bbox_from_mask(mask):
    """Get bounding box (y1, y2, x1, x2) of the masked region"""
    h, w = mask.shape[0], mask.shape[1]
    if mask.sum() < 10:
        return 0, h, 0, w
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return (y1, y2, x1, x2)


def rpb2(src, mask, dst, corner_coord=None, return_type="dict"):
    """Perform Possian Blending using pietorch.blend_c"""
    if isinstance(src, str):
        src = cv2.imread(src).astype(np.uint8)
    if isinstance(mask, str):
        mask = cv2.imread(mask)[:, :, 0] > 128
        mask = mask.astype(np.uint8)
    if isinstance(dst, str):
        dst = cv2.imread(dst).astype(np.uint8)

    bbox = get_bbox_from_mask(mask)

    src = src[:, :, ::-1]  # BGR to RGB
    dst = dst[:, :, ::-1]  # BGR to RGB

    source = TF.to_tensor(src.copy())
    target = TF.to_tensor(dst.copy())
    mask = torch.tensor(mask).float()

    y1, y2, x1, x2 = bbox
    ylen = y2 - y1 + 1
    xlen = x2 - x1 + 1

    if corner_coord is None:
        corner_coord = torch.tensor([random.randint(0, 50), random.randint(0, 50)])

    blending = pietorch.blend_c(
        target, source, mask, corner_coord, False, channels_dim=0, bbox=bbox
    )

    new_mask = torch.zeros_like(target[0])
    new_mask[
        corner_coord[0] : corner_coord[0] + ylen,
        corner_coord[1] : corner_coord[1] + xlen,
    ] = mask[y1 : y2 + 1, x1 : x2 + 1]

    new_mask3 = torch.stack([new_mask] * 3, 0)
    fg_blending = blending * new_mask3

    blending_result = (1.0 - torch.stack([mask] * 3, 0)) * source
    blending_result[:, y1 : y2 + 1, x1 : x2 + 1] += fg_blending[
        :,
        corner_coord[0] : corner_coord[0] + ylen,
        corner_coord[1] : corner_coord[1] + xlen,
    ]

    result_dict = {
        "blending": np.clip(blending.permute(1, 2, 0).numpy() * 255, 0, 255).astype(
            np.uint8
        )[:, :, ::-1],
        "output": np.clip(
            blending_result.permute(1, 2, 0).numpy() * 255, 0, 255
        ).astype(np.uint8)[:, :, ::-1],
    }
    return result_dict


def get_paths(path) -> Dict[str, str]:
    """Parse file paths to retrieve corresponding ground truth, mask, and composite image paths"""
    parts = path.split("/")
    img_name_parts = parts[-1].split(".")[0].split("_")
    if len(img_name_parts) > 3:
        img_name_parts[1] = img_name_parts[0] + "_" + img_name_parts[1]
        img_name_parts.pop(0)

    if "masks" in path:
        base = os.path.join(*parts[:-2])
        name = img_name_parts[0]
        return {
            "gt_path": os.path.join(base, "real_images", f"{name}.jpg"),
            "mask_path": path,
            "image_path": os.path.join(base, "real_images", f"{name}.jpg"),
        }

    elif "composite" in path:
        base = os.path.join(*parts[:-2])
        name, idx = img_name_parts[0], img_name_parts[1]
        return {
            "gt_path": os.path.join(base, "real_images", f"{name}.jpg"),
            "mask_path": os.path.join(base, "masks", f"{name}_{idx}.png"),
            "image_path": path,
        }

    else:
        raise ValueError(f"Unknown path type: {path}")


def cpt_hotmap(img1, img2, mask3):
    """Generate color-coded heatmap showing differences in masked regions"""
    img1 = img1.astype(np.uint8)
    img2 = cv2.resize(img2.astype(np.uint8), (img1.shape[1], img1.shape[0]))
    error_map = cv2.absdiff(img1, img2) * 5
    error_map = error_map * mask3
    heatmap = cv2.applyColorMap(error_map.astype(np.uint8), cv2.COLORMAP_JET)
    return heatmap


if __name__ == "__main__":
    # === Configurable Paths ===
    BASE_DIR = "/path/to/iHarmony4/HFlickr/composite_images"
    SAVE_DIR = "/path/to/save_outputs"
    real_path = BASE_DIR.replace("composite_images", "real_images")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # === Selected composite images to test ===
    select = [
        "f12_1_1.jpg",
        "f14_1_1.jpg",
        "f155_1_1.jpg",
        "f45_1_2.jpg",
        "f70_1_1.jpg",
    ]
    real_images = os.listdir(real_path)

    for i, name in enumerate(select):
        print(f"Processing {i}: {name}")
        paths = get_paths(os.path.join(BASE_DIR, name))

        src = cv2.imread(paths["gt_path"]).astype(np.uint8)
        mask = cv2.imread(paths["mask_path"])[:, :, 0] > 128
        mask = mask.astype(np.uint8)
        mask3 = np.stack([mask] * 3, axis=-1)

        dst = cv2.imread(os.path.join(real_path, choice(real_images))).astype(np.uint8)
        color_transfer = cv2.imread(paths["image_path"]).astype(np.uint8)

        result = rpb2(src, mask, dst, return_type="dict")
        blending = result["blending"]
        output = result["output"]

        scale = uniform(0.6, 0.9)
        output = (
            scale * output * mask3 + (1 - scale) * src * mask3 + src * (1.0 - mask3)
        ).astype(np.uint8)

        hotmap = cpt_hotmap(output, src, mask3)
        hotmap1 = cpt_hotmap(color_transfer, src, mask3)

        # Create subdirectories
        for sub in [
            "dst",
            "comp",
            "blending",
            "hotmap",
            "hotmap1",
            "real",
            "color_transfer",
        ]:
            os.makedirs(os.path.join(SAVE_DIR, sub), exist_ok=True)

        cv2.imwrite(os.path.join(SAVE_DIR, "dst", f"img_{i}.png"), dst)
        cv2.imwrite(os.path.join(SAVE_DIR, "comp", f"img_{i}.png"), output)
        cv2.imwrite(os.path.join(SAVE_DIR, "blending", f"img_{i}.png"), blending)
        cv2.imwrite(os.path.join(SAVE_DIR, "hotmap", f"img_{i}.png"), hotmap)
        cv2.imwrite(os.path.join(SAVE_DIR, "hotmap1", f"img_{i}.png"), hotmap1)
        cv2.imwrite(os.path.join(SAVE_DIR, "real", f"img_{i}.png"), src)
        cv2.imwrite(
            os.path.join(SAVE_DIR, "color_transfer", f"img_{i}.png"), color_transfer
        )
