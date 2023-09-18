import torch
from torchsummary import summary

from dataset.temimagenet import get_temimagenet_trainval
from models.SwinIR import SwinIR
from train.train import sr_train
from configs import training_config

if __name__ == '__main__':
    model = SwinIR(upscale=2, img_size=(256, 256), in_chans=1,
                   window_size=8, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='denoising').cuda()
    summary(model, input_size=(1, 256, 256))
    train_loader, val_loader = get_temimagenet_trainval()
    sr_train(model, train_loader, val_loader, training_config)
