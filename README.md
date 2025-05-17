# Region-to-Region  

Official implementation of the paper **"Region-to-Region: Enhancing Generative Image Harmonization with Adaptive Regional Injection"**. 

**Note**: 
1. This repository is released solely for **anonymous review purposes**. The paper is currently under review.   
2. To ensure a **double-blind** review process, the issue section has been disabled during the review period.

---

## Installation
```bash
conda create -n r2r python=3.10
conda activate r2r
cd Region_to_Region
conda env create -f environment.yml
# pip install -r requirements.txt
```

## Download Checkpoints

Download LDM checkpoint: [here](https://huggingface.co/1243asdad/region2region/tree/main/stable-diffusion-inpainting)  
Download Clear-VAE checkpoint: [here](https://huggingface.co/1243asdad/region2region/tree/main/clear_vae)

Place them under `inference/checkpoints`. forming the following structure:

```shell
inference/
└── checkpoints/
    ├── clear_vae/
    ├── stable-diffusion-inpainting/
```

## Dataset Preparation

### iHarmony4

Please refer to the [DiffHarmony](https://github.com/nicecv/DiffHarmony) to prepare data.
Place the iHarmony4 dataset inside the data folder, forming the following structure:

```shell
data/iHarmony4
|- HAdobe5k/
    |- composite_images
    |- masks
    |- real_images
    |- ...
|- HCOCO/
|- HFlickr/
|- Hday2night/
|- train.jsonl
|- test.jsonl
```

### RPHarmony

Please download it from [here](https://huggingface.co/1243asdad/region2region/blob/main/RPHarmony.zip) and place it under the data folder.
Make sure the structure is just like that:

```shell
data/RPHarmony
|- R-ADE20K
    |- composite_images
    |- masks
    |- real_images
    |- ...
|- R-DUTS
|- train.jsonl
|- test.jsonl
|- train.txt
|- test.txt
```

The RPHarmony dataset is built using our proposed **Random Poisson Blending** method.   
The implementation of Poisson Blending is based on [pytorch-poisson-image-editing](https://github.com/matt-baugh/pytorch-poisson-image-editing).

## Inference

```bash
cd inference
```

For using pretrained checkpoints, see the **Download Checkpoints** section for instructions.  

Test the model performance on the iHarmony4 dataset:

```bash
sh scripts/inference.sh
```

To test on other datasets, simply change the parameters in the `inference.sh`.
You can customize the testing settings by editing the `inference.sh` script.

## Train



The weights of our model are derived from [DiffHarmony](https://github.com/nicecv/DiffHarmony).  
We offer [converted weights](https://huggingface.co/1243asdad/region2region/blob/main/diff-base.ckpt) compatible with ControlNet-style training.  

Please download the pretrained model weights and place them in the `./ckpt` directory. 
Alternatively, you can manually convert the weights using the `tool_add_control.py` 

Please modify the configuration files in the `./configs`, such as `datasets.yaml`, and then:

```bash
sh train.sh
```

For detailed training options and configurations, see `train_control.py`.

### Train Clear-VAE

The VAE is initialized with weights from Stable Diffusion. Please download the pretrained VAE weights from [here](https://huggingface.co/1243asdad/region2region/blob/main/sd_vae.ckpt) and place them in the `./ckpt` directory.

```bash
python train_VAE.py
```

For detailed training options and configurations, see `train_VAE.py`.

## Model weights convert

If you use the weights obtained from training, please run `inference/convert.py` to convert them from Stable Diffusion (SD) format to Diffusers format.  
For parameter settings, please refer to the `convert.py`.

```bash
cd inference
python convert.py  --ckpt_path *** --save_path *** --vae_path *** --vae_save_path ***
```

The `convert.py` provides weight conversion for the core components (e.g., ControlNet, clear-VAE). 
However, you still need to download the remaining weights (e.g., safety_checker) from [here](https://huggingface.co/1243asdad/region2region/tree/main/stable-diffusion-inpainting).

## VAE inference

Download our pretrained Clear-VAE checkpoint (Stable Diffusion format) from [here](https://huggingface.co/1243asdad/region2region/blob/main/clear_vae.ckpt), modify the ckpt_path in `configs/vae.yaml`, and then run:

```bash
python infer_VAE.py
```


## Acknowledgements
This project is developped on the codebase of [ControlNet](https://github.com/lllyasviel/ControlNet), [AnyDoor](https://github.com/ali-vilab/AnyDoor), [DiffHarmony](https://github.com/nicecv/DiffHarmony) and [FinetuneVAE-SD](https://github.com/Leminhbinh0209/FinetuneVAE-SD). We appreciate these great work! 
