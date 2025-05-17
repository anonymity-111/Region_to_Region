import os
import cv2
import einops
import numpy as np
import torch
from ldm.models.autoencoder import AutoencoderKL
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from ihd_dataset import DatasetClearVAE


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f"Loaded model config from [{config_path}]")
    return model


config = OmegaConf.load("./configs/vae.yaml")
device = torch.device("cuda:0")


def get_vae_weights(input_path):
    pretrained_weights = torch.load(input_path)
    if "state_dict" in pretrained_weights:
        pretrained_weights = pretrained_weights["state_dict"]
    vae_weight = {}
    for k in pretrained_weights.keys():
        if "first_stage_model" in k:
            vae_weight[k.replace("first_stage_model.", "")] = pretrained_weights[k]
    return vae_weight


class vae_fine:
    def __init__(self, config):

        embed_dim = config.params.embed_dim
        monitor = config.params.monitor
        ddconfig = config.params.ddconfig
        lossconfig = config.params.lossconfig
        # 初始化 AutoencoderKL 模型
        self.vae_model = AutoencoderKL(
            embed_dim=embed_dim,
            monitor=monitor,
            ddconfig=ddconfig,
            lossconfig=lossconfig, 
        )

        vae_state_dict = torch.load(ddconfig.ckpt_path, map_location=device)
        self.vae_model.load_state_dict(vae_state_dict, strict=False)
        self.vae_model = self.vae_model.to(device)
        self.vae_model.eval()

        self.scale_factor = 0.18215

        for param in self.vae_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.vae_model.encode(x)

    @torch.no_grad()
    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
        return self.scale_factor * z

    @torch.no_grad()
    def decode_first_stage(self, z, skip_connect_hs=None):
        z = 1.0 / self.scale_factor * z
        return self.vae_model.decode(z)

    @torch.no_grad()
    def differentiable_decode_first_stage(
        self, z, predict_cids=False, force_not_quantize=False, skip_connect_hs=None
    ):
        z = 1.0 / self.scale_factor * z
        if self.vae_model.enable_decoder_cond_lora:
            return self.vae_model.decode(z, hs=skip_connect_hs)
        else:
            return self.vae_model.decode(z)


model = vae_fine(config)


def tensor2img(x):
    # img
    if x.shape[1] == 3:
        return (einops.rearrange(x, "b c h w -> b h w c") * 127.5 + 127.5).cpu().numpy()
    # mask
    else:
        return x.cpu().numpy()


def img2tensor(x):
    x = torch.from_numpy(x).float().cuda() / 127.5 - 1.0
    x = einops.rearrange(x, "k h w c -> k c h w")
    return x


def save_img(x_samples, save_file_name, idx):
    torch.cuda.empty_cache()
    x_samples = (
        (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
        .cpu()
        .numpy()
    ) 

    result = x_samples[0][:, :, ::-1]
    result = np.clip(result, 0, 255)

    cv2.imwrite(save_file_name, result)


def to_tensor(item, key):
    tensor = item[key]

    tensor = einops.rearrange(tensor, "b h w c -> b c h w").clone().float().to(device)

    return tensor


def infer(data, idx, file_name, folder):
    if isinstance(file_name, list):
        file_name = file_name[0]

    if not os.path.exists(folder):
        os.mkdir(folder)

    ref = to_tensor(data, "ref")
    tar = to_tensor(data, "tar")

    with torch.no_grad():
        encoder_posterior = model.vae_model.encode(tar)
        z0_tar = encoder_posterior.sample()
        _, skip_connect_ref, _ = model.vae_model.encode_with_cond(ref)
        for idx, skip in enumerate(skip_connect_ref):
            skip_connect_ref[idx] = model.vae_model.filters[idx](skip)
        pred = model.vae_model.decode(z0_tar, hs=skip_connect_ref)

        save_img(pred.float(), os.path.join(folder, file_name), idx)


# Datasets
DConf = OmegaConf.load("./configs/datasets.yaml")

dataset1 = DatasetClearVAE(**DConf.Test.Hday2night)
dataset2 = DatasetClearVAE(**DConf.Test.HFlickr)
dataset3 = DatasetClearVAE(**DConf.Test.HAdobe5k)
dataset4 = DatasetClearVAE(**DConf.Test.HCOCO)
dataset = ConcatDataset([dataset1, dataset2, dataset3, dataset4])


dataloader = DataLoader(dataset, num_workers=8, batch_size=1, shuffle=False)
print("len dataloader: ", len(dataloader))
pbar = tqdm(total=len(dataloader))

for idx, data in enumerate(dataloader):
    pbar.update(1)
    infer(
        data,
        idx + 1,
        file_name=data["save_file"],
        folder="./test",
    )

pbar.close()
print("over!")
