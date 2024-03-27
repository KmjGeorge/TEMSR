import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class ZSSRNet(nn.Module):
    def __init__(self, input_channels=3, kernel_size=3, channels=64, sr_factor=2):
        super(ZSSRNet, self).__init__()

        self.conv0 = nn.Conv2d(input_channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv6 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        # self.conv7 = nn.Conv2d(channels, input_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.upsample = nn.PixelShuffle(sr_factor)
        self.conv7 = nn.Conv2d(channels // (sr_factor ** 2), input_channels, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.upsample(x)
        x = self.conv7(x)

        return x


if __name__ == '__main__':
    from torchsummary import summary

    model = ZSSRNet(input_channels=1, kernel_size=3, channels=64).cuda()
    summary(model, (1, 64, 64))
