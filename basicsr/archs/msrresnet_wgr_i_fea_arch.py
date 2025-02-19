import torch
from torch import nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY
import functools

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


@ARCH_REGISTRY.register()
class MSRResNet_wGR_i_fea(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4, rfea_layer='RB16'):
        super(MSRResNet_wGR_i_fea, self).__init__()
        print('Model: MSRResNet_wGR_i (return feature)')
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)
        self.recon_trunk1 = make_layer(basic_block, nb // 4)
        self.recon_trunk2 = make_layer(basic_block, nb // 4)
        self.recon_trunk3 = make_layer(basic_block, nb // 4)
        self.recon_trunk4 = make_layer(basic_block, nb // 4)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 1:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last], 0.1)
        if self.upscale == 4:
            initialize_weights(self.upconv2, 0.1)

        self.rfea_layer = rfea_layer
        print('Return feature layer: {}'.format(self.rfea_layer))

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        fea_out_Conv1 = fea.view(1, -1)
        out = self.recon_trunk1(fea)
        fea_out_RB4 = out  # .view(1,-1)
        out = self.recon_trunk2(out)
        fea_out_RB8 = out  # .view(1,-1)
        out = self.recon_trunk3(out)
        fea_out_RB12 = out  # .view(1,-1)
        out = self.recon_trunk4(out)
        fea_out_RB16 = out  # .view(1,-1)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            fea_out_UP1 = out.view(1, -1)
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out
        # if self.rfea_layer == 'Conv1':
        #     return out, fea_out_Conv1
        # elif self.rfea_layer == 'RB4':
        #     return out, fea_out_RB4
        # elif self.rfea_layer == 'RB8':
        #     return out, fea_out_RB8
        # elif self.rfea_layer == 'RB12':
        #     return out, fea_out_RB12
        # elif self.rfea_layer == 'RB16':
        #     return out, fea_out_RB16
        # elif self.rfea_layer == 'UP1':
        #     return out, fea_out_UP1


if __name__ == '__main__':
    from torchsummary import summary
    model = MSRResNet_wGR_i_fea(in_nc=1, out_nc=1, upscale=4).cuda()
    summary(model, (1, 64, 64))

