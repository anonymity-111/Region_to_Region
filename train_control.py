import argparse
import os
import pytorch_lightning as pl
from cldm.hack import disable_verbosity, enable_sliced_attention
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from ihd_dataset import IhdDatasetMultiRes
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="train cldm")
parser.add_argument("--ckpt_name", type=str, default="diff-base")
parser.add_argument("--continue_train", default=False, action="store_true")
parser.add_argument("--save_name", type=str, default="test")
parser.add_argument("--max_epochs", type=int, default=75)
parser.add_argument("--sd_locked", type=bool, default=True)
parser.add_argument("--config", type=str, default="cldm_v15_cn")
parser.add_argument("--dataset", type=str, default="ihar")
args = parser.parse_args()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

save_memory = True
disable_verbosity()
if save_memory:
    enable_sliced_attention()

# Configs
resume_path = f"./ckpt/{args.ckpt_name}.ckpt"


batch_size = 4
logger_freq = 5000
learning_rate = 1e-5

sd_locked = args.sd_locked
only_mid_control = False

n_gpus = 2
accumulate_grad_batches = 32 // batch_size

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(f"./configs/{args.config}.yaml").cpu()
if not args.continue_train:
    model.load_state_dict(load_state_dict(resume_path, location="cpu"), strict=False)

model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Datasets
DConf = OmegaConf.load("./configs/datasets.yaml")

if args.dataset == "ihar":
    dataset = IhdDatasetMultiRes(**DConf.Train.iHarmony4)
elif args.dataset == "rph":
    dataset = IhdDatasetMultiRes(**DConf.Train.RPH)
elif args.dataset == "cc":
    dataset = IhdDatasetMultiRes(**DConf.Train.CC)
else:
    exit(0)


# The ratio of each dataset is adjusted by setting the __len__
dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True)
logger = ImageLogger(
    batch_frequency=logger_freq, log_dir_name="image_log_" + f"{args.save_name}"
)
trainer = pl.Trainer(
    gpus=[5,6,7],
    strategy="ddp_sharded",
    precision=16,
    accelerator="gpu",
    callbacks=[logger],
    progress_bar_refresh_rate=1,
    accumulate_grad_batches=accumulate_grad_batches,
    max_epochs=args.max_epochs,
    enable_checkpointing=False,
)

# Train!
if not args.continue_train:
    trainer.fit(model, dataloader)
else:
    print(f"from >> {resume_path} << Load checkpoint, and continue training")
    trainer.fit(model, dataloader, ckpt_path=resume_path)

trainer.save_checkpoint(
    f"./ckpt/{args.save_name}.ckpt"
)
