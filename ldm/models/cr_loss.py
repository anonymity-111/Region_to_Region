import random
import einops
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg16
import ldm.models.pietorch as pietorch
import cv2
from PIL import Image
from torchvision.transforms import functional as TF

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


class CR_loss(nn.Module):
    def __init__(self, k=3) -> None:
        super().__init__()

        self.D = nn.L1Loss()
        self.k = k
        VGG16_model = vgg16(pretrained=True)
        self.SR_extractor = VGG16_model.features[:23]

    def forward(self, pred, gt_image, mask, composition=None, return_neg_samp=False):
        """
        composition (N, 3, H, W):           composition image (input image to the network)
        pred        (N, 3, H, W):           predicted image (output of the network)
        gt_image    (N, 3, H, W):           ground truth image
        mask        (N, 1, H, W):           image mask
        """

        with torch.no_grad():
            neg_samp = create_negative_samples(
                gt_image, mask[:, 0, :, :]
            )

            if return_neg_samp:
                return neg_samp

            # calculate foreground and background style representations
            self.SR_extractor = self.SR_extractor.to(pred.get_device())
            self.SR_extractor.eval()

            f = self.SR_extractor(pred * mask)
            f_plus = self.SR_extractor(gt_image * mask)
            f_minus = [self.SR_extractor(neg_samp[:, k] * mask) for k in range(self.k)]

            l_ss_cr = self.D(f, f_plus) / (
                self.D(f, f_plus)
                + torch.sum(torch.tensor([self.D(f, f_minus_k) for f_minus_k in f_minus]))
                + 1e-8
            )

        return l_ss_cr

    def get_loss(self, pred, gt_image, mask, composition=None, return_neg_samp=False):
        """
        composition (N, 3, H, W):           composition image (input image to the network)
        pred        (N, 3, H, W):           predicted image (output of the network)
        gt_image    (N, 3, H, W):           ground truth image
        mask        (N, 1, H, W):           image mask
        """
        with torch.no_grad():

            # calculate foreground and background style representations
            self.SR_extractor = self.SR_extractor.to(pred.get_device())
            self.SR_extractor.eval()

            f = self.SR_extractor(pred * mask)
            f_plus = self.SR_extractor(gt_image * mask)
            f_minus = self.SR_extractor(composition * mask)

            l_ss_cr = self.D(f, f_plus) / (self.D(f, f_plus) + self.D(f, f_minus) + 1e-8)

            return l_ss_cr

    # def gen_neg_data(self, src, dst, mask_01):
    #     mask3 = np.stack([mask_01, mask_01, mask_01], -1)
    #     mask = 255 * mask_01

    #     mean_color = cv2.mean(src, mask=mask)

    #     solid_bg = np.full(src.shape, mean_color[:3], dtype=np.uint8)

    #     _src = src * mask3 + (1.0 - mask3) * solid_bg

    #     kernel = np.ones((3, 3), np.uint8)

    #     _mask = cv2.dilate(mask, kernel, iterations=1)[: mask.shape[0], : mask.shape[1]]

    #     y1, y2, x1, x2 = get_bbox_from_mask(mask)

    #     center = (math.ceil((x1 + x2) / 2), math.ceil((y1 + y2) / 2))
    #     output = cv2.seamlessClone(src, dst, _mask, center, cv2.NORMAL_CLONE).astype(
    #         np.uint8
    #     )

    #     scale = 0.75
    #     output = (
    #         scale * output * mask3 + (1 - scale) * src * mask3 + src * (1.0 - mask3)
    #     )

    #     return output[:, :, ::-1]

def create_negative_samples(gt_images, masks,k=3):
    """
    samples K-1 negative samples for each composited image based on the other images in the batch

    gt_images   (N, 3, H, W):           ground truth images
    mask        (N, 1, H, W):           image mask

    neg_samp    (N, K, 3, H, W):        K negative samples
    """

    N, C, H, W = gt_images.shape


    device = gt_images.device
    gt_images = [gt_images[i] for i in range(N)]
    masks = [masks[i].squeeze(0) for i in range(N)]

    neg_samp = []

    for i in range(N):
        neg_samp_i = [
            gen_neg_data(
                gt_images[i],
                gt_images[random.randint(0, N - 1)],
                masks[i],
            ).to(device)
            for _ in range(k)
        ]

        neg_samp_i = torch.stack(neg_samp_i, dim=0)


        neg_samp.append(neg_samp_i)

    return torch.stack(neg_samp, dim=0)

def gen_neg_data(src: torch.Tensor, dst: torch.Tensor, mask: torch.Tensor):
    """
    Generate a negative image sample using PyTorch tensor inputs.

    Args:
        src: Tensor, shape (3, H, W), float32, range [0,1], source image
        dst: Tensor, shape (3, H, W), float32, range [0,1], destination image
        mask: Tensor, shape (H, W), float32 or uint8, values in {0,1} or {0,255}

    Returns:
        np.ndarray: BGR image, uint8
    """
    device = src.device
    if mask.dtype != torch.float32:
        mask = (mask > 0).to(torch.float32)

    y1, y2, x1, x2 = get_bbox_from_mask(mask.cpu().numpy())
    ylen = y2 - y1 + 1
    xlen = x2 - x1 + 1

    corner_coord = torch.tensor(
        [
            random.randint(0, max(0, src.shape[1] - ylen - 1)),
            random.randint(0, max(0, src.shape[2] - xlen - 1)),
        ]
    )

    blending = pietorch.blend_c(
        dst.detach().cpu(),
        src.detach().cpu(),
        mask.detach().cpu(),
        corner_coord.detach().cpu(),
        False,
        channels_dim=0,
        bbox=(y1, y2, x1, x2),
    ).to(device)

    # Create new mask in target
    new_mask = torch.zeros_like(dst[0])
    new_mask[
        corner_coord[0] : corner_coord[0] + ylen,
        corner_coord[1] : corner_coord[1] + xlen,
    ] = mask[y1 : y2 + 1, x1 : x2 + 1]

    mask3 = mask.expand(3, -1, -1)
    new_mask3 = new_mask.expand(3, -1, -1)

    fg_blending = blending * new_mask3
    composed = (1.0 - mask3) * src
    composed[:, y1 : y2 + 1, x1 : x2 + 1] += fg_blending[
        :,
        corner_coord[0] : corner_coord[0] + ylen,
        corner_coord[1] : corner_coord[1] + xlen,
    ]

    # Convert to BGR uint8 numpy
    composed_np = composed.permute(1, 2, 0).cpu().numpy() * 255
    composed_np = np.clip(composed_np, 0, 255).astype(np.uint8)[:,:,::-1]
        
    return composed.clamp(0,1)



class Perceptual_Loss(nn.Module):
    def __init__(
        self,
    ):
        super(Perceptual_Loss, self).__init__()
        VGG16_model = vgg16(pretrained=True)
        self.SR_extractor = VGG16_model.features[:23]
        self.D = nn.L1Loss()

    def forward(self, pred, real, mask=None):
        self.SR_extractor = self.SR_extractor.to(pred.get_device())
        self.SR_extractor.eval()

        if mask is None:
            return self.D(self.SR_extractor(pred), self.SR_extractor(real))
        else:
            return self.D(
                self.SR_extractor(pred * mask), self.SR_extractor(real * mask)
            )


def get_bbox_from_mask(mask):
    h, w = mask.shape[0], mask.shape[1]

    if mask.sum() < 10:
        return 0, h, 0, w
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return (y1, y2, x1, x2)

