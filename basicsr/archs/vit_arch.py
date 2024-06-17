# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import numpy as np
import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed, Block


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, mode='pool', **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.mode = mode
        if self.mode == 'pool':
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], -1, h * p, h * p))
        return imgs

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.mode == 'pool':
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        elif self.mode == 'cls':
            x = self.norm(x)  # only cls token
            outcome = x[:, 0]
        elif self.mode == 'map':
            outcome = x[:, 1:]
            outcome = self.unpatchify(outcome)
        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        return x


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch16_gray(**kwargs):
    model = VisionTransformer(in_chans=1,
                              patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


if __name__ == '__main__':
    from torchsummary import summary
    import cv2
    import matplotlib.pyplot as plt

    mean = 0.2187
    std = 0.1075
    model = vit_base_patch16_gray(mode='map').cuda()
    model.load_state_dict(torch.load(r'F:\Project\mae-main\output_dir\mae_base(DF)\checkpoint-99.pth')['model'],
                          strict=False)
    x = cv2.imread(r'D:\Datasets\STEMEXP\all_DF\warwick_DF\img352 (2)_s003.png', 0).astype(float)
    x = cv2.resize(x, (224, 224)) / 255.0
    x = (x - mean) / std
    x = torch.from_numpy(x[np.newaxis, np.newaxis, ...])
    x = x.float().cuda()
    y = model(x).detach().cpu().squeeze(0)
    y = torch.permute(y, (1, 2, 0)).numpy()
    print(y.shape, y.max(), y.min())
    plt.imshow(y, 'gray')
    plt.show()

    summary(model, input_size=(1, 224, 224))
