import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from math import exp

import torchvision.transforms
from torch.autograd import Variable
import torch.nn as nn
from torchvision.models import vgg19, vgg16_bn
import pytorch_ssim
import torchvision.transforms as ttf
import lpips
import configs
from models.AtomSegNet import UNet


def cal_psnr(img1, img2):
    """img1, img2: 0~1"""
    return 10 * torch.log10(1 / F.mse_loss(img1, img2))


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size / 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel:
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class PixelwiseLoss(nn.Module):
    def forward(self, inputs, targets):
        return F.smooth_l1_loss(inputs, targets)


class FeatureLoss(nn.Module):
    def __init__(self, loss, blocks, weights, device):
        super().__init__()
        self.feature_loss = loss
        assert all(isinstance(w, (int, float)) for w in weights)
        assert len(weights) == len(blocks)

        self.weights = torch.tensor(weights).to(device)
        # VGG16 contains 5 blocks - 3 convolutions per block and 3 dense layers towards the end
        assert len(blocks) <= 5
        assert all(i in range(5) for i in blocks)
        assert sorted(blocks) == blocks

        vgg = vgg16_bn(pretrained=True).features
        vgg.eval()

        for param in vgg.parameters():
            param.requires_grad = False

        vgg = vgg.to(device)

        bns = [i - 2 for i, m in enumerate(vgg) if isinstance(m, nn.MaxPool2d)]
        assert all(isinstance(vgg[bn], nn.BatchNorm2d) for bn in bns)

        self.hooks = [FeatureHook(vgg[bns[i]]) for i in blocks]
        self.features = vgg[0: bns[blocks[-1]] + 1]

    def forward(self, inputs, targets):

        # normalize foreground pixels to ImageNet statistics for pre-trained VGG
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        trans = torchvision.transforms.Normalize(mean, std)
        inputs = trans(inputs)
        targets = trans(targets)

        # extract feature maps
        self.features(inputs)
        input_features = [hook.features.clone() for hook in self.hooks]

        self.features(targets)
        target_features = [hook.features for hook in self.hooks]

        loss = 0.0

        # compare their weighted loss
        for lhs, rhs, w in zip(input_features, target_features, self.weights):
            lhs = lhs.view(lhs.size(0), -1)
            rhs = rhs.view(rhs.size(0), -1)
            loss += self.feature_loss(lhs, rhs) * w

        return loss


class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.on)

    def on(self, module, inputs, outputs):
        self.features = outputs

    def close(self):
        self.hook.remove()


def perceptual_loss(x, y):
    F.mse_loss(x, y)


def PerceptualLoss(blocks, weights, device):
    return FeatureLoss(perceptual_loss, blocks, weights, device)


class TopologyAwareLoss(nn.Module):
    def __init__(self, criteria, weights):
        # Here criteria -> [PixelwiseLoss, PerceptualLoss],
        # weights -> [1, mu] (or any other combination weights)
        assert len(weights) == len(criteria)

        self.criteria = criteria
        self.weights = weights

    def forward(self, inputs, targets):
        loss = 0.0
        for criterion, w in zip(self.criteria, self.weights):
            each = w * criterion(inputs, targets)
            loss += each

        return loss


def cal_ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(configs.device)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def atom_loss_fn(weight_path='./weights/circularMask.pth'):
    unet = UNet()
    unet.load_state_dict(torch.load(weight_path))
    unet = unet.eval().to(configs.device)
    return unet


def cal_atom_loss(atom_loss_fn, img1, img2):
    # 转为单通道
    img1_gray = torch.mean(img1, dim=1, keepdim=True)
    img2_gray = torch.mean(img2, dim=1, keepdim=True)
    img1_heatmap = atom_loss_fn(img1_gray)
    img2_heatmap = atom_loss_fn(img2_gray)

    # print(img1_heatmap)
    # print(img2_heatmap)
    # plt.subplot(121)
    # plt.imshow(img1_heatmap[0].detach().squeeze_().cpu().numpy())
    # plt.subplot(122)
    # plt.imshow(img2_heatmap[0].detach().squeeze_().cpu().numpy())
    # plt.show()
    loss = F.smooth_l1_loss(img1_heatmap, img2_heatmap)
    return loss


if __name__ == '__main__':
    from PIL import Image

    im1 = np.array(Image.open('../00295.png').convert('L'))
    im2 = np.array(Image.open('../00310.png').convert('L'))
    im1 = im1[np.newaxis, np.newaxis, ...]
    im2 = im2[np.newaxis, np.newaxis, ...]
    im1 = (torch.from_numpy(im1).float() / 255.0).cuda()
    im2 = (torch.from_numpy(im2).float() / 255.0).cuda()
    ssim = cal_ssim(im1, im2)
    psnr = cal_psnr(im1, im2)
    lpips_fn = lpips.LPIPS(net='vgg').cuda()
    print('ssim=', ssim.item())
    print('psnr=', psnr.item())
    print('lpips=', lpips_fn(im1, im2).item())
