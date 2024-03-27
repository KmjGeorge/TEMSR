import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
from basicsr.utils.registry import ARCH_REGISTRY


# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# SE
class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


# Channel MLP: Conv1*1 -> Conv1*1
class ChannelMLP(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.mlp(x)


# MBConv: Conv1*1 -> DW Conv3*3 -> [SE] -> Conv1*1
class MBConv(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.mbconv = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim),
            nn.GELU(),
            SqueezeExcitation(hidden_dim),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.mbconv(x)


# CCM
class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.ccm(x)


# SAFM
class SAFM(nn.Module):
    def __init__(self, dim, n_levels=3):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])

        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out


class FreBlock(nn.Module):
    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(nc, nc, 1, 1, 0))

    def forward(self, x):
        mag = torch.abs(x)
        pha = torch.angle(x)
        mag = self.processmag(mag)
        pha = self.processpha(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        return x_out


class SFT(nn.Module):
    def __init__(self, nc):
        super(SFT, self).__init__()
        self.convmul = nn.Conv2d(nc, nc, 3, 1, 1)
        self.convadd = nn.Conv2d(nc, nc, 3, 1, 1)
        self.convfuse = nn.Conv2d(2 * nc, nc, 1, 1, 0)

    def forward(self, x, res):
        # res = res.detach()
        mul = self.convmul(res)
        add = self.convadd(res)
        fuse = self.convfuse(torch.cat([x, mul * x + add], 1))
        return fuse


class FreBlockAdjust(nn.Module):
    def __init__(self, nc):
        super(FreBlockAdjust, self).__init__()
        self.processmag = nn.Conv2d(nc, nc, 1, 1, 0)

        self.processpha = nn.Conv2d(nc, nc, 1, 1, 0)
        self.sft = SFT(nc)
        self.cat = nn.Conv2d(2 * nc, nc, 1, 1, 0)

    def forward(self, x, y_amp, y_phase):
        mag = torch.abs(x)
        pha = torch.angle(x)
        mag = self.processmag(mag)
        pha = self.processpha(pha)
        mag = self.sft(mag, y_amp)
        pha = self.cat(torch.cat([y_phase, pha], 1))
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)

        return x_out



class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        # Multiscale Block
        self.safm = SAFM(dim)
        # Feedforward layer
        self.ccm = CCM(dim, ffn_scale)

    def forward(self, x):
        x = self.safm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x


class FreqFeature(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.down = nn.AvgPool2d(2)
        self.freqprocess = FreBlock(dim)

    def forward(self, x):
        x = self.down(x)
        x_freq = torch.fft.rfft2(x, norm='backward')
        x_freq = self.freqprocess(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, norm='backward')
        x = F.interpolate(x_freq_spatial, scale_factor=2, mode='nearest')
        return x


class FinalAdjust(nn.Module):
    def __init__(self, in_chans, out_chans, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adjust = FreBlockAdjust(in_chans)
        self.fuse = nn.Conv2d(in_chans * 2, out_chans, 1, 1)

    def forward(self, x, y_amp, y_phase):
        x_freq = self.adjust(x, y_amp, y_phase)
        x_freq_spatial = torch.fft.irfft2(x_freq)
        x = self.fuse(torch.cat([x_freq_spatial, x_freq]))
        return x


@ARCH_REGISTRY.register()
class SAFMN_FFT(nn.Module):
    def __init__(self, in_chans, dim, n_blocks=8, ffn_scale=2.0, upscaling_factor=4):
        super().__init__()
        self.to_feat = nn.Conv2d(in_chans, dim, 3, 1, 1)

        self.feats = nn.ModuleList([AttBlock(dim, ffn_scale) for _ in range(n_blocks)])
        self.feats_fft = nn.ModuleList([FreqFeature(dim) for _ in range(n_blocks)])
        self.fuse = nn.ModuleList([nn.Conv2d(dim * 2, dim, 1, 1) for _ in range(n_blocks)])
        if upscaling_factor != 1:
            self.to_img = nn.Sequential(
                nn.Conv2d(dim, in_chans * upscaling_factor ** 2, 3, 1, 1),
                nn.PixelShuffle(upscaling_factor)
            )
        else:
            self.to_img = nn.Conv2d(dim, in_chans, 3, 1, 1)

    def forward(self, x):
        x = self.to_feat(x)
        for i in range(len(self.feats)):
            spatial_feat = self.feats[i](x)
            freq_feat = self.feats_fft[i](x)
            # print(spatial_feat.shape, freq_feat.shape)
            x = self.fuse[i](torch.cat([spatial_feat, freq_feat], dim=1))
        x_up = self.to_img(x)

        return x_up


if __name__ == '__main__':
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    from torchsummary import summary

    # x = torch.randn(1, 3, 640, 360)
    # x = torch.randn(1, 3, 427, 240)
    x = torch.randn(1, 4, 128, 128).cuda()
    # x = torch.randn(1, 3, 256, 256)

    model = SAFMN_FFT(in_chans=1, dim=60, n_blocks=12, ffn_scale=2.0, upscaling_factor=1).cuda()
    # model = SAFMN(dim=36, n_blocks=12, ffn_scale=2.0, upscaling_factor=2)
    # print(model)
    summary(model, input_size=(1, 256, 256))
    # print('\n\n')
    # print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    # print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    # output = model(x)
    # print(output.shape)
