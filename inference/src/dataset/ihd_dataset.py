import json
import os
import os.path
from argparse import Namespace
from typing import Dict, List

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


def get_paths(path) -> Dict[str, str]:
    parts = path.split("/")
    img_name_parts = parts[-1].split(".")[0].split("_")
    if len(img_name_parts) > 3:
        img_name_parts[1] = img_name_parts[0] + "_" + img_name_parts[1]
        img_name_parts.pop(0)
    if "masks" in path:
        return {
            "gt_path": os.path.join(
                *parts[:-2], "real_images", f"{img_name_parts[0]}.jpg"
            ),
            "mask_path": path,
            "image_path": os.path.join(
                *parts[:-2], "real_images", f"{img_name_parts[0]}.jpg"
            ),
        }
    elif "composite" in path:
        return {
            "gt_path": os.path.join(
                "/", *parts[:-2], "real_images", f"{img_name_parts[0]}.jpg"
            )
            if "data1" in path
            else os.path.join(*parts[:-2], "real_images", f"{img_name_parts[0]}.jpg"),
            "mask_path": os.path.join(
                "/",
                *parts[:-2],
                "masks",
                f"{img_name_parts[0]}_{img_name_parts[1]}.png",
            )
            if "data1" in path
            else os.path.join(
                *parts[:-2], "masks", f"{img_name_parts[0]}_{img_name_parts[1]}.png"
            ),
            "image_path": path,
        }
    else:
        raise ValueError(f"Unknown path type: {path}")


class IhdDatasetMultiRes(Dataset):
    def __init__(self, split, tokenizer, resolutions: List[int], opt):
        self.image_paths = []
        self.captions = []
        self.split = split
        self.tokenizer = tokenizer
        self.resolutions = list(set(resolutions))
        self.random_flip = opt.random_flip
        self.random_crop = opt.random_crop
        self.mask_dilate = opt.mask_dilate

        data_file = opt.train_file if split == "train" else opt.test_file
        if split == "test":
            self.random_flip = False
            self.random_crop = False

        with open(os.path.join(opt.dataset_root, data_file), "r") as f:
            for line in f:
                cont = json.loads(line.strip())
                image_path = os.path.join(
                    opt.dataset_root,
                    cont["file_name"],
                )
                self.image_paths.append(image_path)
                self.captions.append(cont.get("text", ""))

        self.create_image_transforms()

    def __len__(self):
        return len(self.image_paths)

    def create_image_transforms(self):
        self.rgb_normalizer = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

    def CannyDetector(self, image, mask):
        image, mask = np.array(image).astype(np.uint8), np.array(mask).astype(np.uint8)
        mask3 = np.stack([mask] * 3, -1)

        fg = mask3 * image + np.ones_like(image) * 255 * (1 - mask3)
        gray_image = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
        # 应用Canny边缘检测
        canny_map = cv2.Canny(gray_image, 100, 200)  # 100和200是低阈值和高阈值
        canny_map = cv2.cvtColor(canny_map, cv2.COLOR_GRAY2BGR)

        res = (canny_map * mask3 + (1.0 - mask3) * image).astype(np.uint8)

        return Image.fromarray(res)

    def fg_extract(self, image, mask):
        image, mask = np.array(image).astype(np.uint8), np.array(mask).astype(np.uint8)
        kernel = np.ones((10, 10), np.uint8)
        mask_ = cv2.dilate(mask, kernel, iterations=1)[: mask.shape[0], : mask.shape[1]]

        mask3 = np.stack([mask_] * 3, -1)

        fg = mask3 * image + np.ones_like(image) * 255 * (1 - mask3)

        return Image.fromarray(fg)

    def __getitem__(self, index):
        paths = get_paths(self.image_paths[index])

        try:
            comp = Image.open(paths["image_path"]).convert("RGB")  # RGB , [0,255]
        except SyntaxError:
            print("comp:")
            print(paths["image_path"])
            exit(0)
        try:
            mask = Image.open(paths["mask_path"]).convert("1")
        except SyntaxError:
            print("mask:")
            print(paths["mask_path"])
            exit(0)
        try:
            real = Image.open(paths["gt_path"]).convert("RGB")  # RGB , [0,255]
        except SyntaxError:
            print("real:")
            print(paths["gt_path"])
            exit(0)


        caption = self.captions[index]
        
        if self.tokenizer is not None:
            caption_ids = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids[0]
        else:
            caption_ids = torch.empty(size=(1,), dtype=torch.long)

        if self.random_flip and np.random.rand() > 0.5 and self.split == "train":
            comp, mask, real = TF.hflip(comp), TF.hflip(mask), TF.hflip(real)
        if self.random_crop:
            for _ in range(5):
                mask_tensor = TF.to_tensor(mask)
                crop_box = T.RandomResizedCrop.get_params(
                    mask_tensor, scale=[0.5, 1.0], ratio=[3 / 4, 4 / 3]
                )
                cropped_mask_tensor = TF.crop(mask_tensor, *crop_box)
                h, w = cropped_mask_tensor.shape[-2:]
                if cropped_mask_tensor.sum() < 0.01 * h * w:
                    continue
                break

        ref = self.fg_extract(comp, mask)
        canny = self.CannyDetector(comp, mask)

        example = {}
        for resolution in self.resolutions:
            if self.random_crop:
                this_res_comp = TF.resize(
                    TF.crop(comp, *crop_box),
                    size=(resolution, resolution),
                )  # default : bilinear resample ; antialias for PIL Image
                this_res_real = TF.resize(
                    TF.crop(real, *crop_box),
                    size=(resolution, resolution),
                )  # default : bilinear resample ; antialias for PIL Image
                this_res_mask = TF.resize(
                    TF.crop(mask, *crop_box),
                    size=(resolution, resolution),
                )  # default : bilinear resample ; antialias for PIL Image
                this_res_canny = TF.resize(
                    TF.crop(canny, *crop_box),
                    size=(resolution, resolution),
                )  # default : bilinear resample ; antialias for PIL Image
            else:
                if comp.size == (resolution, resolution):
                    this_res_comp = comp
                    this_res_mask = mask
                    this_res_real = real
                    this_res_canny = canny

                else:
                    this_res_comp = TF.resize(
                        comp, [resolution, resolution]
                    )  # default : bilinear resample ; antialias for PIL Image
                    this_res_mask = TF.resize(
                        mask, [resolution, resolution]
                    )  # default : bilinear resample ; antialias for PIL Image
                    this_res_real = TF.resize(
                        real, [resolution, resolution]
                    )  # default : bilinear resample ; antialias for PIL Image
                    this_res_canny = TF.resize(
                        canny, [resolution, resolution]
                    )  # default : bilinear resample ; antialias for PIL Image

            this_res_comp = self.rgb_normalizer(this_res_comp)  # tensor , [-1,1]
            this_res_canny = self.rgb_normalizer(this_res_canny)
            this_res_real = self.rgb_normalizer(this_res_real)  # tensor , [-1,1]
            this_res_mask = TF.to_tensor(this_res_mask)
            this_res_mask = (this_res_mask >= 0.5).float()  # mask : tensor , 0/1
            this_res_mask = self.dilate_mask_image(this_res_mask)

            # print(this_res_canny.shape)
            # print(this_res_comp.shape)
            # print(this_res_mask.shape)

            example[resolution] = {
                "real": this_res_real,
                "mask": this_res_mask,
                "comp": this_res_comp,
                # 'ref':ref,
                "hint": torch.cat([this_res_canny, this_res_comp, this_res_mask], 0),
                "real_path": paths["gt_path"],
                "mask_path": paths["mask_path"],
                "comp_path": paths["image_path"],
                "caption": caption,
                "caption_ids": caption_ids,
            }

        return example

    def dilate_mask_image(self, mask: torch.Tensor) -> torch.Tensor:
        if self.mask_dilate > 0:
            mask_np = (mask * 255).numpy().astype(np.uint8)
            mask_np = cv2.dilate(
                mask_np, np.ones((self.mask_dilate, self.mask_dilate), np.uint8)
            )
            mask = torch.tensor(mask_np.astype(np.float32) / 255.0)
        return mask


class IhdDatasetSingleRes(Dataset):
    def __init__(self, split, tokenizer, resolution, opt):
        self.resolution = resolution
        self.multires_ds = IhdDatasetMultiRes(split, tokenizer, [resolution], opt)

    def __len__(self):
        return len(self.multires_ds)

    def __getitem__(self, index):
        return self.multires_ds[index][self.resolution]


subset_names = [
    "HAdobe5k",
    "HCOCO",
    "Hday2night",
    "HFlickr",
]


def extract_ds_name(path):
    for subset_name in subset_names:
        if subset_name in path:
            return subset_name
    return None


def read_jsonl_file(filename):
    data = []
    with open(filename, "r") as file:
        for line in file:
            data.append(json.loads(line))
    return data


SUBSET_TO_MIX = [
    "HCOCO",
    "HFlickr",
    "HAdobe5k",
    "Hday2night",
]


def Img2array(item):
    return np.array(item).astype(np.uint8)
