import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from basicsr.utils.registry import ARCH_REGISTRY
from functools import partial


@ARCH_REGISTRY.register()
class MAEIR(nn.Module):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_num_heads=16, decoder_depth=8,
                 mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()

        self.in_chans = in_chans
        # --------------------------------------------------------------------------
        # Encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        self.init_parameters()

    def init_parameters(self):  # train decoder only
        for key, para in self.named_parameters():
            if 'decoder' in key:
                para.requires_grad = True
            else:
                para.requires_grad = False

    def forward_encoder(self, x):
        # encoder
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_decoder(self, x):
        x = self.decoder_embed(x)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # (196, 256)
        return x

    def forward(self, x):
        with torch.no_grad():
            x = self.forward_encoder(x)
        x = self.forward_decoder(x)
        return x

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 * C)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs


if __name__ == '__main__':
    model = MAEIR(img_size=224, patch_size=16,
                  in_chans=1,
                  embed_dim=1024,
                  depth=24,
                  num_heads=16,
                  decoder_embed_dim=384,
                  decoder_num_heads=16,
                  decoder_depth=4,
                  mlp_ratio=4).cuda()
    from torchsummary import summary
    summary(model, input_size=(1, 224, 224))
