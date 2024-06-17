import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.NAFNet_arch import NAFBlock
from basicsr.utils.registry import ARCH_REGISTRY

class ICB(nn.Module):
    """
    Instruction Condition Block (ICB)
    Paper Section 3.3
    """

    def __init__(self, feature_dim, text_dim=768):
        super(ICB, self).__init__()
        self.fc = nn.Linear(text_dim, feature_dim)
        self.block = NAFBlock(feature_dim)
        self.beta = nn.Parameter(torch.zeros((1, feature_dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, feature_dim, 1, 1)), requires_grad=True)

    def forward(self, x, text_embedding):
        gating_factors = torch.sigmoid(self.fc(text_embedding))
        gating_factors = gating_factors.unsqueeze(-1).unsqueeze(-1)

        f = x * self.gamma + self.beta  # 1) learned feature scaling/modulation
        f = f * gating_factors  # 2) (soft) feature routing based on text
        f = self.block(f)  # 3) block feature enhancement
        return f + x

@ARCH_REGISTRY.register()
class InstructIR(nn.Module):
    """
    InstructIR model using NAFNet (ECCV 2022) as backbone.
    The model takes as input an RGB image and a text embedding (encoded instruction).
    Described in Paper Section 3.3
    """

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], txtdim=768):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.enc_cond = nn.ModuleList()
        self.dec_cond = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

            self.enc_cond.append(ICB(chan, txtdim))

            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            # Add text embedding as modulation
            self.dec_cond.append(ICB(chan, txtdim))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, txtembd):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        encs = []

        for encoder, enc_mod, down in zip(self.encoders, self.enc_cond, self.downs):
            x = encoder(x)
            x = enc_mod(x, txtembd)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip, dec_mod in zip(self.decoders, self.ups, encs[::-1], self.dec_cond):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            x = dec_mod(x, txtembd)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


if __name__ == '__main__':
    from torchsummary import summary

    model = InstructIR(img_channel=1, width=32, enc_blk_nums=[1, 1, 1, 8], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1],
                       txtdim=512).cuda()
    summary(model, input_size=[(1, 256, 256), (512,)])  # Total params: 9,501,025  32 1118 1 1111
                                                        # Total params: 30,879,649 32 2228 12 2222
