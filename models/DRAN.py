import torch.nn as nn
import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F

import configs

torch_ver = torch.__version__[:3]


class HFIRM(nn.Module):
    def __init__(self, channel=3, nf=64, scale=4):
        super(HFIRM, self).__init__()
        self.scale = scale
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # self.upsample = F.upsample(size=(h, w), mode='bilinear')
        self.conv1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        # self.res = nn.Sequential(*[nn.Conv2d(nf, nf, 3,  1, 1, bias=True),
        #                                   nn.ReLU(inplace=True),
        #                                   nn.Conv2d(nf, nf, 3,  1, 1, bias=True)])
        self.resblock = ResBlock(conv=default_conv, n_feats=nf, kernel_size=3, bias=True, bn=False, act=nn.ReLU(True),
                                 res_scale=1)
        self.body = nn.Sequential(*[self.resblock for _ in range(3)])
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.avg_pool(x)
        y = F.interpolate(input=y, size=(h, w), mode='nearest')
        # y = self.upsample(y)
        y = self.avg_pool(y)
        y = F.interpolate(input=y, size=(h, w), mode='nearest')
        # y = y-x
        y = torch.clamp(x - y, min=0, max=255.0)
        out = self.conv1(y)
        out = self.body(out)
        out = self.conv2(out)
        return out


class HighFrequencyExtractionBlock11(nn.Module):
    def __init__(self, channel=3, nf=64, scale=4):
        super(HighFrequencyExtractionBlock11, self).__init__()
        self.scale = scale
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # self.upsample = F.upsample(size=(h, w), mode='bilinear')
        self.conv = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.res = nn.Sequential(*[nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(nf, nf, 3, 1, 1, bias=True)])
        # self.resblock = ResBlock(conv=default_conv(), n_feats=nf, kernel_size=3, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.avg_pool(x)
        y = self.avg_pool(y)
        y = F.interpolate(input=y, size=(h, w), mode='nearest')
        out = self.conv(y)
        out = self.res(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class PAM_Module(Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2) * dilation, bias=bias, dilation=(dilation, dilation))


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y + x


class SALayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SALayer, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv_du = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size, padding=padding, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size, padding=padding, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        # x = torch.cat([avg_out, max_out], dim=1)
        y = self.conv_du(avg_out)
        return x * y + x


class RG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size=3, n_resblocks=10):
        super(RG, self).__init__()
        module_body = [ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) for _ in
                       range(n_resblocks)]

        module_body.append(conv(n_feat, n_feat, kernel_size))
        self.modules_body = nn.Sequential(*module_body)
        # self.ca = ChannelAttention(in_planes=n_feat)
        # self.sa = SpatialAttention()
        # self.ca = CAM_Module(n_feat)
        # self.sa = PAM_Module(n_feat)

    def forward(self, x):
        residual = x
        res = self.modules_body(x)

        res = 0.2 * res + residual

        return res


class DRAN(nn.Module):
    def __init__(self, n_feats=64, n_resblocks=24, n_resgroups=24, scale=4, n_colors=3, conv=default_conv):
        super(DRAN, self).__init__()

        n_feat = n_feats
        self.n_blocks = n_resblocks
        self.n_resgroups = n_resgroups

        kernel_size = 3
        scale = scale
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(255.0, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(n_colors, n_feat, kernel_size)]
        # modules_head = [nn.Conv2d(n_colors, n_feat, 5, 1, 2)]

        # modules_head_2 = [conv(n_feat, n_feat, kernel_size)]
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1, padding=0, bias=True),

        # define body module
        self.modules_body = nn.ModuleList(
            [RG(conv, n_feat, kernel_size, self.n_blocks) for _ in range(self.n_resgroups)])

        modules_tail = [
            conv(n_feat, n_feat, kernel_size),
            Upsampler(conv, scale, n_feat, act=False)
        ]
        self.conv = conv(n_feat, n_colors, kernel_size)
        self.add_mean = MeanShift(255, rgb_mean, rgb_std, 1)

        self.head_1 = nn.Sequential(*modules_head)
        # self.head_2 = nn.Sequential(*modules_head_2)
        self.fusion = nn.Sequential(*[nn.Conv2d(n_feat * 2, n_feat, 1, padding=0, stride=1)])
        self.fusion_end = nn.Sequential(*[nn.Conv2d(n_feat * self.n_resgroups, n_feat, 1, padding=0, stride=1)])

        # self.body = nn.Sequential(*self.modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.hfeb = HFIRM(nf=n_feats)

        self.ca = CALayer(channel=n_feat)
        self.sa = SALayer(kernel_size=7)

    def forward(self, x):

        # x = self.sub_mean(x)
        first = x
        x = self.head_1(x)
        # print(x[0][0])
        res = x
        # x = self.head_2(x)

        res_x = x

        fusions = []
        for i, l in enumerate(self.modules_body):
            figure = x
            x = l(x)
            fusions.append(self.fusion(torch.cat((self.sa(x), self.ca(x)), 1)))
            # fusions.append(x)
        # y = self.fusion_end(torch.cat(fusions, 1))
        y = self.ca(self.fusion_end(torch.cat(fusions, 1)))
        hfeb = self.hfeb(first)
        res = res + self.sa(x) + hfeb + y
        # res = self.sa(x) + res + hfeb
        # res = x + y + res
        x = self.tail(res)
        x = self.conv(x)
        # x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


# 原始输入为48*48
def get_dran():
    model = DRAN(**configs.dran_config)
    return model
