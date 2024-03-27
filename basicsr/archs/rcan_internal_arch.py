import torch
from torch import nn as nn
from torch.nn.utils import spectral_norm

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import Upsample, make_layer


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True), nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, squeeze_factor=16, res_scale=1):
        super(RCAB, self).__init__()
        self.res_scale = res_scale

        self.rcab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ReLU(True), nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor))

    def forward(self, x):
        res = self.rcab(x) * self.res_scale
        return res + x


class RCABwithAFT(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, squeeze_factor=16, res_scale=1):
        super(RCABwithAFT, self).__init__()
        self.res_scale = res_scale

        self.rcab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1).requires_grad_(False),
            nn.ReLU(True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1).requires_grad_(False),
            ChannelAttention(num_feat, squeeze_factor).requires_grad_(False),
            nn.Conv2d(num_feat, num_feat, 1, 1))

        nn.init.zeros_(self.rcab[4].weight)
        # self.scale_block = nn.Sequential(
        #     spectral_norm(nn.Conv2d(num_feat, num_feat // 2, 1, 1)),
        #     nn.LeakyReLU(0.2, True),
        #     spectral_norm(nn.Conv2d(num_feat // 2, num_feat, 1, 1)))
        # self.shift_block = nn.Sequential(
        #     spectral_norm(nn.Conv2d(num_feat, num_feat // 2, 1, 1)),
        #     nn.LeakyReLU(0.2, True),
        #     spectral_norm(nn.Conv2d(num_feat // 2, num_feat, 1, 1)),
        #     nn.Sigmoid())
        # for module in self.scale_block:
        #     if not isinstance(module, nn.LeakyReLU):
        #         nn.init.zeros_(module.weight)
        # for module in self.shift_block:
        #     if not (isinstance(module, nn.LeakyReLU) or isinstance(module, nn.Sigmoid)):
        #         nn.init.zeros_(module.weight)

    def forward(self, x):
        # scale = self.scale_block(x)
        # shift = self.shift_block(x)
        # sft_res = self.rcab(x) * scale + shift
        # res = sft_res * self.res_scale
        res = self.rcab(x) * self.res_scale
        return res + x


class ResidualGroup(nn.Module):
    """Residual Group of RCAB.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_block (int): Block number in the body network.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, num_block, squeeze_factor=16, res_scale=1):
        super(ResidualGroup, self).__init__()

        self.residual_group = make_layer(
            RCABwithAFT, num_block, num_feat=num_feat, squeeze_factor=squeeze_factor, res_scale=res_scale)
        self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        res = self.conv(self.residual_group(x))
        return res + x


# class ResidualGroupWithAFT(nn.Module):
#     def __init__(self, num_feat, num_block, squeeze_factor=16, res_scale=1):
#         super(ResidualGroupWithAFT, self).__init__()
#
#         self.residual_group = make_layer(
#             RCAB, num_block, num_feat=num_feat, squeeze_factor=squeeze_factor, res_scale=res_scale).requires_grad_(
#             False)
#         self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1).requires_grad_(False)
#         self.aft_1 = nn.Conv2d(num_feat, num_feat * 2, 3)
#         self.aft_2 = nn.Conv2d(num_feat*2, num_feat, 3)
#         # self.conv_beta = nn.Conv2d(num_feat, 1, 1, 1)
#
#     def forward(self, x):
#         res = self.conv(self.residual_group(x))
#         res_transform = self.aft_2(self.aft_1(res))
#         return res_transform + x


@ARCH_REGISTRY.register()
class RCAN_Internal(nn.Module):
    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_group=10,
                 num_block=16,
                 squeeze_factor=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):  # DIV2K
        super(RCAN_Internal, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1).requires_grad_(False)
        self.body = make_layer(
            ResidualGroup,
            num_group,
            num_feat=num_feat,
            num_block=num_block,
            squeeze_factor=squeeze_factor,
            res_scale=res_scale)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1).requires_grad_(False)
        self.upsample = Upsample(upscale, num_feat).requires_grad_(False)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1).requires_grad_(False)

    def forward(self, x):
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        x = self.body(x)
        res = self.conv_after_body(x)
        res += x

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean

        return x
