import torch
from torchsummary import summary

from dataset.temimagenet import get_temimagenet_trainval
from models.SwinIR import get_swinir
from train.train import sr_train
import configs


if __name__ == '__main__':
    configs.save_config()
    model = get_swinir().to(configs.device)
    summary(model, input_size=(1, 256, 256))
    train_loader, val_loader = get_temimagenet_trainval()
    sr_train(model, train_loader, val_loader, configs.training_config)
