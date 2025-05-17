# https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/README.md
import os
from argparse import ArgumentParser
from contextlib import contextmanager
from datetime import datetime

import einops
import pytorch_lightning as pl
import torch
import torch.optim as optim
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import ConcatDataset

from ihd_dataset import DatasetClearVAE
from ldm.models.autoencoder import AutoencoderKL
from ldm.models.cr_loss import CR_loss


class DataModule(pl.LightningDataModule):
    def __init__(self, train_ds, val_ds, batch_size=64):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch_size = batch_size
        self.setup("fit")

    def setup(self, stage):
        print(f"Train size: {len(self.train_ds)}, Val size: {len(self.val_ds)}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=8
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=8
        )


class FinetuneVAE(pl.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        momentum=0.9,
        weight_decay=5e-4,
        config=None,
        vae_weights=None,
        device=torch.device("cuda"),
        ema_decay=-0.999,
        precision=32,
        num_epochs=2,
        log_dir=None,
        enable_cr_loss=True,
        enable_p_loss=False,
        enable_filters=True,
    ):
        super().__init__()

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.hfp_weight = 10
        self.cr_loss = CR_loss()

        embed_dim = config.params.embed_dim
        monitor = config.params.monitor
        ddconfig = config.params.ddconfig
        lossconfig = config.params.lossconfig

        self.model = AutoencoderKL(
            embed_dim=embed_dim,
            monitor=monitor,
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            ema_decay=ema_decay,
            tune=True,
        )

        self.enable_cr_loss = enable_cr_loss
        self.enable_p_loss = enable_p_loss
        self.enable_filters = enable_filters

        if vae_weights is not None:
            self.model.load_state_dict(vae_weights, strict=False)

        for param in self.model.encoder.parameters():
            param.requires_grad = False
        for param in self.model.decoder.parameters():
            param.requires_grad = True

        self.precision = precision
        self.log_dir = log_dir
        self.log_one_batch = False
        self.use_ema = ema_decay > 0

        self.init_save = True
        self.num_epochs = num_epochs
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        params = list(self.model.decoder.parameters())
        if self.enable_filters:
            params += list(self.model.filters.parameters())
        optimizer = optim.AdamW(params, lr=self.lr)

        return optimizer

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Assuming the DataModule is attached to the Trainer and accessible
            self.train_ds = self.trainer.datamodule.train_ds
            self.val_ds = self.trainer.datamodule.val_ds
            print("Warning: The setup method is called")

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        ref, tar, mask = batch["ref"], batch["tar"], batch["mask"]
        ref = einops.rearrange(ref, "b h w c -> b c h w")
        tar = einops.rearrange(tar, "b h w c -> b c h w")
        mask = einops.rearrange(mask, "b h w c -> b c h w")
        if self.precision == 16:
            ref = ref.half()
            tar = tar.half()
            mask = mask.half()

        with torch.no_grad():
            encoder_posterior = self.model.encode(tar)
            z0_tar = encoder_posterior.sample()
            _, skip_connect_ref, _ = self.model.encode_with_cond(ref)

        if self.enable_filters:
            for idx, skip in enumerate(skip_connect_ref):
                skip_connect_ref[idx] = self.model.filters[idx](skip)

        pred = self.model.decode(z0_tar, hs=skip_connect_ref)
        rec_loss = torch.abs(tar.contiguous() - pred.contiguous()).mean()

        loss = rec_loss.clone()
        self.log(
            "rec_loss",
            rec_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )

        if self.enable_cr_loss:
            cr_loss = 0.3 * self.cr_loss.get_loss(
                pred.contiguous(),
                tar.contiguous(),
                mask.contiguous(),
                composition=ref.contiguous(),
            )
            # cr_loss = 0.3 * self.cr_loss(
            #     pred.contiguous(),
            #     tar.contiguous(),
            #     mask.contiguous(),
            #     composition=ref.contiguous(),
            # )
            loss += cr_loss.clone()
            self.log(
                "cr_loss",
                cr_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=False,
            )
        if self.enable_p_loss:
            p_loss = self.perceptual_loss(
                pred.contiguous(), tar.contiguous(), mask.contiguous()
            )
            loss += p_loss.clone()
            self.log(
                "p_loss",
                p_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=False,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        ref, tar, mask = batch["ref"], batch["tar"], batch["mask"]
        ref = einops.rearrange(ref, "b h w c -> b c h w")
        tar = einops.rearrange(tar, "b h w c -> b c h w")
        mask = einops.rearrange(mask, "b h w c -> b c h w")
        if self.precision == 16:
            ref = ref.half()
            tar = tar.half()
            mask = mask.half()

        with torch.no_grad():
            encoder_posterior = self.model.encode(tar)
            z0_tar = encoder_posterior.sample()
            torch.cuda.empty_cache()

            _, skip_connect_ref, _ = self.model.encode_with_cond(ref)
            # z0_ref = cond_posterior.sample()
            torch.cuda.empty_cache()

            if self.enable_filters:
                for idx, skip in enumerate(skip_connect_ref):
                    skip_connect_ref[idx] = self.model.filters[idx](skip)
            torch.cuda.empty_cache()

            pred = self.model.decode(z0_tar, hs=skip_connect_ref)
            torch.cuda.empty_cache()

            rec_loss = torch.abs(tar.contiguous() - pred.contiguous()).mean()
            loss = rec_loss
            self.log(
                "rec_loss",
                rec_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=False,
            )
        return {"val_loss": loss, "rec_loss": rec_loss}

    def validation_epoch_end(self, validation_step_outputs):
        self.log_one_batch = False
        val_loss = torch.stack([x["val_loss"] for x in validation_step_outputs]).mean()
        rec_loss = torch.stack([x["rec_loss"] for x in validation_step_outputs]).mean()

        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "val_rec_loss",
            rec_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        if (self.trainer.current_epoch + 1) == self.trainer.max_epochs:
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.log_dir, f"model_{self.trainer.current_epoch + 1}.ckpt"
                ),
            )


def get_vae_weights(input_path):
    pretrained_weights = torch.load(input_path)
    if "state_dict" in pretrained_weights:
        pretrained_weights = pretrained_weights["state_dict"]
    vae_weight = {}
    for k in pretrained_weights.keys():
        if "first_stage_model" in k:
            vae_weight[k.replace("first_stage_model.", "")] = pretrained_weights[k]
    del pretrained_weights
    return vae_weight


def argument_inputs():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--sd_vae_path", type=str, default="./ckpt/sd_vae.ckpt")
    parser.add_argument("--enable_cr_loss", type=bool, default=True)
    parser.add_argument("--enable_p_loss", type=bool, default=False)
    parser.add_argument("--enable_filters", type=bool, default=True)
    parser.add_argument("--output_dir",type=str,default="./vae_finetune",)
    parser.add_argument("--note",type=str,default="test",)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_inputs()
    # device = torch.device("cuda:7")
    file_names = (
        datetime.now().strftime("%Y-%m-%d-%H:%M")
        + f"_bs({args.batch_size})_lr({args.lr})_epochs({args.num_epochs})"
    )
    log_dir = f"{args.output_dir}/{file_names}"
    os.makedirs(log_dir, exist_ok=True)

    vae_config = OmegaConf.load("./configs/vae.yaml")

    # input_path = "./ckpt/v2-1_512-ema-pruned.ckpt"
    # vae_weight = get_vae_weights(input_path)
    # torch.save("./ckpt/sd_vae.ckpt")

    vae_weight = torch.load(args.sd_vae_path)

    DConf = OmegaConf.load("./configs/datasets.yaml")

    val_ds = DatasetClearVAE(**DConf.Test.Hday2night)

    dataset1 = DatasetClearVAE(**DConf.Train.HAdobe5k)
    dataset2 = DatasetClearVAE(**DConf.Train.HCOCO)
    dataset3 = DatasetClearVAE(**DConf.Train.Hday2night)
    dataset4 = DatasetClearVAE(**DConf.Train.HFlickr)

    train_ds = ConcatDataset([dataset1, dataset2, dataset3, dataset4])

    data_module = DataModule(
        train_ds=train_ds,
        val_ds=val_ds,
        batch_size=args.batch_size,
    )

    model = FinetuneVAE(
        config=vae_config,
        vae_weights=vae_weight,
        lr=args.lr,
        log_dir=log_dir,
        enable_cr_loss=args.enable_cr_loss,
        enable_filters=args.enable_filters,
        enable_p_loss=args.enable_p_loss
    )

    trainer = Trainer(
        min_epochs=1,
        max_epochs=args.num_epochs,
        precision=16,
        strategy="ddp_sharded",
        gpus=[4, 5],
        num_sanity_val_steps=0,
        accumulate_grad_batches=16 // args.batch_size,
        default_root_dir=log_dir,
        enable_checkpointing=True,
    )

    trainer.fit(model, datamodule=data_module)

    os.makedirs('./ckpt',exist_ok=True)
    trainer.save_checkpoint(f"./ckpt/VAE_{args.note}.ckpt")
