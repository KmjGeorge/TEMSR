import torch
from models.UHDFour import InteractNet
from torchsummary import summary

UHDFour = InteractNet().cuda()
x_sample = torch.rand(1, 1, 256, 256)
summary(UHDFour, input_size=(3, 256, 256))
