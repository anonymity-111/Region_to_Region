import torch
from omegaconf import OmegaConf
from src.util.convert_from_ckpt import (
    convert_controlnet_checkpoint,
    convert_ldm_clip_checkpoint,
    convert_ldm_unet_checkpoint,
    convert_ldm_vae_checkpoint,
    create_unet_diffusers_config,
    create_vae_diffusers_config,
)
from safetensors.torch import save_file
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default="../ckpt/Diff-test.ckpt")
parser.add_argument("--convert_cn", type=bool,default=True, )
parser.add_argument("--convert_unet",type=bool,default=True,)
parser.add_argument("--convert_clip",type=bool,default=True,)
parser.add_argument("--save_path",type=str,default="./checkpoints/stable-diffusion-inpainting")
parser.add_argument("--vae_path",type=str,default="..ckpt/clear_vae.ckpt",)
parser.add_argument("--convert_vae",type=bool,default=True)
parser.add_argument("--vae_save_path",type=str,default="./checkpoints/clear_vae1/diffusion_pytorch_model.safetensors")
args = parser.parse_args()


DConf = OmegaConf.load("../configs/cldm_v15_cn.yaml")

unet_config = create_unet_diffusers_config(DConf, image_size=32, controlnet=False)
controlnet_config = create_unet_diffusers_config(DConf, image_size=32, controlnet=True)
vae_config = create_vae_diffusers_config(DConf, image_size=32)

# OmegaConf.save(unet_config, "unet.yaml")
# with open("config.json", "w") as f:
#     json.dump(controlnet_config, f, indent=1)
# OmegaConf.save(vae_config, "vae.yaml")

if args.convert_vae:
    assert args.vae_path is not None
    print("convert vae from :"+args.vae_path)
    
    # VAE
    VAE_checkpoint = torch.load(
        args.vae_path,
        map_location="cpu",
    )
    if "state_dict" in VAE_checkpoint.keys():
        VAE_checkpoint = VAE_checkpoint["state_dict"]
    VAEConf = vae_config
    vae_weight = convert_ldm_vae_checkpoint(
        checkpoint=VAE_checkpoint, config=VAEConf
    )
    save_file(
        vae_weight,
        args.vae_save_path,
    )

    print("vae weight convert over")

print("convert ldm from: " + args.ckpt_path)
checkpoint = torch.load(args.ckpt_path)

if "state_dict" in checkpoint.keys():
    checkpoint = checkpoint["state_dict"]

if args.convert_unet:
    
    unet_weight = convert_ldm_unet_checkpoint(
        checkpoint=checkpoint, config=DConf, path=None
    )

    torch.save(
        unet_weight,
        os.path.join(args.save_path, "unet/diffusion_pytorch_model.bin"),
    )

if args.convert_cn:
    _, cn_weight = convert_controlnet_checkpoint(
        checkpoint=checkpoint,
        original_config=DConf,
        checkpoint_path=None,
        image_size=32,
        upcast_attention=False,
        extract_ema=False,
    )
    torch.save(
        cn_weight,
        os.path.join(args.save_path,"controlnet/diffusion_pytorch_model.bin"),
    )

if args.convert_clip:
    
    clip_weight = convert_ldm_clip_checkpoint(checkpoint=checkpoint)
    torch.save(
        clip_weight,
        os.path.join(args.save_path,"text_encoder/pytorch_model.bin"),
    )

print("convert weights (from sd to diffusers) over!")
