import json
import os
import os.path
import re
from typing import Dict
import cv2
import einops
import numpy as np
import PIL.Image
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
    def __init__(self, split, opt, resolution=512, tokenizer=None):
        # if resolutions is None:
        #     resolution = [512]
        self.image_paths = []
        self.captions = []
        self.split = split
        self.tokenizer = tokenizer
        # self.resolutions = list(set(resolutions))
        self.resolution = (resolution, resolution)
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

    def sample_timestep(self, max_step=1000):
        step_start = 0
        step_end = max_step

        if np.random.rand() < 0.3:
            step_start = 0
            step_end = max_step // 2

        step = np.random.randint(step_start, step_end)
        return np.array([step])

    def create_image_transforms(self):
        self.rgb_normalizer = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

    def CannyDetector(self, image, mask):
        image, mask = np.array(image).astype(np.uint8), np.array(mask).astype(np.uint8)
        mask3 = np.stack([mask] * 3, -1)

        fg = mask3 * image + np.ones_like(image) * 255 * (1 - mask3)
        gray_image = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
        canny_map = cv2.Canny(gray_image, 100, 200)
        canny_map = cv2.cvtColor(canny_map, cv2.COLOR_GRAY2BGR)

        res = (canny_map * mask3 + (1.0 - mask3) * image).astype(np.uint8)

        return PIL.Image.fromarray(res)

    def fg_extract(self, image, mask):
        image, mask = np.array(image).astype(np.uint8), np.array(mask).astype(np.uint8)
        kernel = np.ones((10, 10), np.uint8)
        mask_ = cv2.dilate(mask, kernel, iterations=1)[: mask.shape[0], : mask.shape[1]]

        mask3 = np.stack([mask_] * 3, -1)

        fg = mask3 * image + np.ones_like(image) * 255 * (1 - mask3)

        return PIL.Image.fromarray(fg)

    def rerange(self, tensor):
        return einops.rearrange(tensor, "c h w -> h w c")

    def __getitem__(self, index):
        paths = get_paths(self.image_paths[index])

        comp = Image.open(paths["image_path"]).convert("RGB")  # RGB , [0,255]
        mask = Image.open(paths["mask_path"]).convert("1")
        real = Image.open(paths["gt_path"]).convert("RGB")  # RGB , [0,255]

        caption = self.captions[index]

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

        if self.random_crop:
            this_res_comp = TF.resize(
                TF.crop(comp, *crop_box),
                size=self.resolution,
            )  # default : bilinear resample ; antialias for PIL Image
            this_res_real = TF.resize(
                TF.crop(real, *crop_box),
                size=self.resolution,
            )  # default : bilinear resample ; antialias for PIL Image
            this_res_mask = TF.resize(
                TF.crop(mask, *crop_box),
                size=self.resolution,
            )  # default : bilinear resample ; antialias for PIL Image
            this_res_canny = TF.resize(
                TF.crop(canny, *crop_box),
                size=self.resolution,
            )  # default : bilinear resample ; antialias for PIL Image
        else:
            if comp.size == self.resolution:
                this_res_comp = comp
                this_res_mask = mask
                this_res_real = real
                this_res_canny = canny
            else:
                this_res_comp = TF.resize(
                    comp, list(self.resolution)
                )  # default : bilinear resample ; antialias for PIL Image
                this_res_mask = TF.resize(
                    mask, list(self.resolution)
                )  # default : bilinear resample ; antialias for PIL Image
                this_res_real = TF.resize(
                    real, list(self.resolution)
                )  # default : bilinear resample ; antialias for PIL Image
                this_res_canny = TF.resize(
                    canny, list(self.resolution)
                )  # default : bilinear resample ; antialias for PIL Image

        this_res_ref = self.rgb_normalizer(TF.resize(ref, (224, 224)))
        this_res_canny = self.rgb_normalizer(this_res_canny)
        this_res_comp = self.rgb_normalizer(this_res_comp)  # tensor , [-1,1]
        this_res_real = self.rgb_normalizer(this_res_real)  # tensor , [-1,1]
        this_res_mask = TF.to_tensor(this_res_mask)
        this_res_mask = (this_res_mask >= 0.5).float()  # mask : tensor , 0/1
        this_res_mask = self.dilate_mask_image(this_res_mask)

        this_res_comp, this_res_mask, this_res_real, this_res_canny, this_res_ref = (
            self.rerange(this_res_comp),
            self.rerange(this_res_mask),
            self.rerange(this_res_real),
            self.rerange(this_res_canny),
            self.rerange(this_res_ref),
        )

        example = {
            "ref": this_res_ref,
            "txt": caption,
            "hint": torch.cat([this_res_canny, this_res_comp, this_res_mask], -1),
            "jpg": torch.cat([this_res_real, this_res_comp, this_res_mask], -1),
            "time_steps": self.sample_timestep(),
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


def rerange(tensor):
    return einops.rearrange(tensor, "c h w -> h w c")


class DatasetClearVAE(Dataset):
    def __init__(self, image_dir, mode):
        super().__init__()
        self.image_root = image_dir
        self.mode = mode
        # self.data = os.listdir(self.image_root)

        match = re.search(r"/(?P<content>[^/]+)/[^/]*$", image_dir)
        dataset_name = match.group("content")

        if self.mode == "train":
            data_file = image_dir.replace(
                "/composite_images", f"/{dataset_name}_train.txt"
            )
        else:
            data_file = image_dir.replace(
                "/composite_images", f"/{dataset_name}_test.txt"
            )
        with open(data_file, "r") as f:
            self.data = [line.strip() for line in f]

        self.resolution = (256, 256)
        # if self.mode == 'test':
        #     self.resolution = (512,512)
        self.clip_size = (224, 224)
        self.dynamic = 0

        self.random_flip = False
        self.random_crop = False

        if self.mode == "train":
            self.random_crop = True
            self.random_flip = True

        self.create_image_transforms()

    def create_image_transforms(self):
        self.rgb_normalizer = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _index1 = self.data[idx].find("_")
        _index2 = self.data[idx].find("_", _index1 + 1)
        ref_image_path = os.path.join(self.image_root, self.data[idx])
        tar_image_path = os.path.join(
            self.image_root.replace("/composite_images", "/real_images"),
            self.data[idx][:_index1] + ".jpg",
        )
        mask_path = os.path.join(
            self.image_root.replace("/composite_images", "/masks"),
            self.data[idx][:_index2] + ".png",
        )

        save_file = self.data[idx].replace("jpg", "png")

        comp = Image.open(ref_image_path).convert("RGB")  # RGB , [0,255]
        mask = Image.open(mask_path).convert("1")
        real = Image.open(tar_image_path).convert("RGB")  # RGB , [0,255]

        if self.random_flip and np.random.rand() > 0.5 and self.mode == "train":
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

        if self.random_crop:
            this_res_comp = TF.resize(
                TF.crop(comp, *crop_box),
                size=self.resolution,
            )  # default : bilinear resample ; antialias for PIL Image
            this_res_real = TF.resize(
                TF.crop(real, *crop_box),
                size=self.resolution,
            )  # default : bilinear resample ; antialias for PIL Image
            this_res_mask = TF.resize(
                TF.crop(mask, *crop_box),
                size=self.resolution,
            )  # default : bilinear resample ; antialias for PIL Image
        else:
            if comp.size == self.resolution:
                this_res_comp = comp
                this_res_mask = mask
                this_res_real = real
            else:
                this_res_comp = TF.resize(
                    comp, size=self.resolution
                )  # default : bilinear resample ; antialias for PIL Image
                this_res_mask = TF.resize(
                    mask, size=self.resolution
                )  # default : bilinear resample ; antialias for PIL Image
                this_res_real = TF.resize(
                    real, size=self.resolution
                )  # default : bilinear resample ; antialias for PIL Image

        ori = np.array(this_res_comp)
        this_res_comp = self.rgb_normalizer(this_res_comp)  # tensor , [-1,1]
        this_res_real = self.rgb_normalizer(this_res_real)  # tensor , [-1,1]
        this_res_mask = TF.to_tensor(this_res_mask)
        this_res_mask = (this_res_mask >= 0.5).float()  # mask : tensor , 0/1
        this_res_mask = torch.cat([this_res_mask] * 3, dim=0)

        this_res_comp, this_res_mask, this_res_real = (
            rerange(this_res_comp),
            rerange(this_res_mask),
            rerange(this_res_real),
        )

        example = {
            "tar": this_res_real,
            "mask": this_res_mask,
            "ref": this_res_comp,
            "path": ref_image_path,
            "ori": ori,
        }

        if self.mode == "test":
            example["save_file"] = save_file

        return example
