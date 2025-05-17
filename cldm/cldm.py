import einops
import torch
import torch as th
import torch.nn as nn
from einops import rearrange, repeat
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.attention import (
    MACA,
    SpatialTransformer,
)
from ldm.modules.diffusionmodules.openaimodel import (
    AttentionBlock,
    Downsample,
    ResBlock,
    TimestepEmbedSequential,
    UNetModel,
)
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    timestep_embedding,
    zero_module,
)

# from ldm.modules.dual_encoder_fusion import CrossAttentionInteraction
from ldm.util import exists, instantiate_from_config, log_txt_as_img
from torchvision.utils import make_grid


def count_parameters(model):

    total_params = sum(p.numel() for p in model.parameters())
    total_params_in_M = total_params / 1e6 
    return total_params_in_M


class ControlNet_CA(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        # mask_mask mask_global global
    ):
        super().__init__()

        if use_spatial_transformer:
            assert context_dim is not None, (
                "Fool!! You forgot to include the dimension of your cross-attention conditioning..."
            )

        if context_dim is not None:
            assert use_spatial_transformer, (
                "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            )
            from omegaconf.listconfig import ListConfig

            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, (
                "Either num_heads or num_head_channels has to be set"
            )

        if num_head_channels == -1:
            assert num_heads != -1, (
                "Either num_heads or num_head_channels has to be set"
            )

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            print(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        channel_size = [320] + [640] + [1280] * 2

        self.PCCAs = nn.ModuleList([MACA(channel) for channel in channel_size])

        # self.CA_layers = nn.ModuleList([nn.Identity()])  # channel attention
        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1)),
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        mask_size = 64
        ds = 1
        for level, mult in enumerate(channel_mult):  # 4个stage: 1 2 4 8
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or nr < num_attention_blocks[level]
                    ):
                        # layers.append(PixelwisePixelPixelwiseChannelAttention(ch))
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,
                            )
                        )

                self.input_blocks.append(
                    TimestepEmbedSequential(*layers)
                )  # input_blocks

                self.zero_convs.append(self.make_zero_conv(ch))  # zero_convs
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(  # input_blocks
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )

                ch = out_ch
                input_block_chans.append(ch)

                self.zero_convs.append(self.make_zero_conv(ch))  # zero_convs
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(  # middle_block
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            )
            if not use_spatial_transformer
            else SpatialTransformer(  # always uses a self-attn
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn,
                use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint,
            ),
            # PixelwiseChannelAttention(ch),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))
        )

    def forward(self, x, hint, timesteps, context, mask=None, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)  # 1,1280

        if mask is None:
            _, mask, _ = torch.split(x, [4, 1, 4], dim=1)

        # 1,320,64,64
        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)

        pcca_id = 0

        for idx, (module, zero_conv) in enumerate(
            zip(self.input_blocks, self.zero_convs)
        ):
            if guided_hint is not None:
                # skip the first layer
                h = guided_hint
                guided_hint = None
            else:
                h_new = module(h, emb, context)
                h = h_new

            if hasattr(self, "PCCAs"):
                if (idx % 3 == 0 and idx != 0) or idx == 11:
                    t_mask = torch.nn.functional.interpolate(
                        mask, size=(h.shape[-2], h.shape[-1])
                    )
                    h = self.PCCAs[pcca_id](h, t_mask)

                    pcca_id += 1

            outs.append(zero_conv(h, emb, context))


        h_new = self.middle_block(h, emb, context)

        outs.append(self.middle_block_out(h_new, emb, context))
        return outs


class ControlledUnetModel(UNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        control=None,
        only_mid_control=False,
        mask=None,
        **kwargs,
    ):
        hs = []

        if control is not None:
            with torch.no_grad():
                t_emb = timestep_embedding(
                    timesteps, self.model_channels, repeat_only=False
                )
                emb = self.time_embed(t_emb)
                h = x.type(self.dtype)
                for module in self.input_blocks:  # 1
                    h = module(h, emb, context)
                    hs.append(h)
                h = self.middle_block(h, emb, context)  # 2
        else:
            t_emb = timestep_embedding(
                timesteps, self.model_channels, repeat_only=False
            )
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:  # 1
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)  # 2

        if mask is None:
            _, mask, _ = torch.split(x, [4, 1, 4], dim=1)

        if hasattr(self, "skip_connect"):
            for i, hs_i in enumerate(hs):
                t_mask = torch.nn.functional.interpolate(
                    mask, size=(hs_i.shape[-2], hs_i.shape[-1])
                )

                hs[i] = self.skip_connect[i](hs_i, t_mask)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):  # 3

            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)



class ControlLDM(LatentDiffusion):
    def __init__(
        self,
        control_stage_config,
        control_key,
        only_mid_control,
        iscontrol,
        stage1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if iscontrol:
            self.control_model = instantiate_from_config(control_stage_config)
        else:
            self.control_model = nn.Module()
            print("No ControlNet")

        # self.connect_model =

        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

        self.orig_embeds_params = (
            self.cond_stage_model.transformer.get_input_embeddings().weight.data.clone()
        )

        self.stage1 = stage1
        self.stage2 = False

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):

        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        # get control_image
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, "b h w c -> b c h w")
        control = control.to(memory_format=torch.contiguous_format).float()
        self.time_steps = batch["time_steps"]
        return x, dict(c_crossattn=[c], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, is_control=False, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond["c_crossattn"], 1)


        if cond["c_concat"] is None or is_control == False:
            eps = diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=None,
                only_mid_control=self.only_mid_control,
            )
        else:
            # _, mask,_ = torch.split(x_noisy, [4, 1,4], dim=1)
            mask = None
            if self.channels == 4:
                x_noisy, mask = torch.split(x_noisy, [4, 1], dim=1)
            control = self.control_model(
                x=x_noisy,
                hint=torch.cat(cond["c_concat"], 1),
                timesteps=t,
                context=cond_txt,
                mask=mask,
            )

            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=control,
                only_mid_control=self.only_mid_control,
            )
        return eps

    @torch.no_grad()
    def get_hidden_feature(
        self,
        x_noisy,
        t,
        cond,
        return_encoder=False,
        return_middle=True,
        *args,
        **kwargs,
    ):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond["c_crossattn"], 1)

        if cond["c_concat"] is None:
            hidden_feature = diffusion_model.get_hidden_feature(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=None,
                only_mid_control=self.only_mid_control,
            )
        else:
            control = self.control_model(
                x=x_noisy,
                hint=torch.cat(cond["c_concat"], 1),
                timesteps=t,
                context=cond_txt,
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            hidden_feature = diffusion_model.get_hidden_feature(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=control,
                only_mid_control=self.only_mid_control,
            )
        return hidden_feature

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        ############
        # uncond =  self.get_learned_conditioning([ torch.zeros((1,self.cond_input_channels,224,224)) ] * N)
        # return uncond
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=8,
        n_row=2,
        sample=False,
        ddim_steps=50,
        ddim_eta=0.0,
        return_keys=None,
        quantize_denoised=True,
        inpaint=True,
        plot_denoise_rows=False,
        plot_progressive_rows=True,
        plot_diffusion_rows=False,
        unconditional_guidance_scale=9.0,
        unconditional_guidance_label=None,
        use_ema_scope=True,
        **kwargs,
    ):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        if isinstance(c, list):
            c = self.get_learned_conditioning(c)
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z[:, 0:4, :, :])

        # ==== visualize the shape mask or the high-frequency map ====
        guide_mask = (c_cat[:, -1, :, :].unsqueeze(1) + 1) * 0.5
        guide_mask = torch.cat([guide_mask, guide_mask, guide_mask], 1)
        HF_map = c_cat[:, :3, :, :]  # * 2.0 - 1.0

        log["control"] = HF_map

        log["conditioning"] = log_txt_as_img(
            (512, 512), batch[self.cond_stage_key], size=16
        )

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]

            if self.channels == 9:
                z_start, mask, latent_masked_image = torch.split(
                    z_start, [4, 1, 4], dim=1
                )  #########

            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)

                    if self.channels == 9:
                        input = torch.concatenate(
                            [z_noisy, mask, latent_masked_image]
                        )  #########

                        diffusion_row.append(self.decode_first_stage(input))
                    else:
                        diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, "n b c h w -> b n c h w")
            diffusion_grid = rearrange(diffusion_grid, "b n c h w -> (b n) c h w")
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(
                cond={"c_concat": [c_cat], "c_crossattn": [c]},
                batch_size=N,
                ddim=use_ddim,
                ddim_steps=ddim_steps,
                eta=ddim_eta,
            )
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        unconditional_guidance_scale = 1.1

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(
                z=z,
                cond={"c_concat": [c_cat], "c_crossattn": [c]},
                batch_size=N,
                ddim=use_ddim,
                ddim_steps=ddim_steps,
                eta=ddim_eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc_full,
            )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = (
                x_samples_cfg  # * 2.0 - 1.0
            )
        return log

    @torch.no_grad()
    def sample_log(self, z, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (4, h // 8, w // 8)

        if self.channels == 9:
            latent, mask, masked_latent = torch.split(z, [4, 1, 4], dim=1)
            samples, intermediates = ddim_sampler.sample(
                ddim_steps,
                batch_size,
                shape,
                cond,
                verbose=False,
                x0=torch.cat([latent, masked_latent], dim=1),
                mask=mask,
                **kwargs,
            )
        else:
            latent, mask = torch.split(z, [4, 1], dim=1)
            samples, intermediates = ddim_sampler.sample(
                ddim_steps,
                batch_size,
                shape,
                cond,
                verbose=False,
                x0=latent,
                mask=mask,
                **kwargs,
            )
        return samples, intermediates

    def configure_optimizers(self):
        # assert self.stage1 == False
        lr = self.learning_rate
        params = []
        if not self.stage1:
            params += list(self.control_model.parameters())

            if not self.sd_locked:
                params += list(self.model.diffusion_model.output_blocks.parameters())
                params += list(self.model.diffusion_model.out.parameters())

        else:
            params += list(self.model.diffusion_model.parameters())


        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
