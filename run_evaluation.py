import pandas as pd
import torch
from torchsummary import summary
import lpips

from dataset.temimagenet import get_temimagenet_trainval
from models.SwinIR import get_swinir
from train.train import validate_epoch
import configs

if __name__ == '__main__':
    model = get_swinir().to(configs.device)
    # summary(model, input_size=(1, 256, 256))
    train_loader, val_loader = get_temimagenet_trainval()

    savename = 'uhdfour_3loss'
    epoch_list = []
    val_loss_list = []
    val_psnr_list = []
    val_ssim_list = []
    val_lpips_list = []
    for epoch in range(1, 10):
        model.load_state_dict(torch.load('weights/{}_epoch{}.pt'.format(savename, epoch)))
        val_loss, val_psnr, val_ssim, val_lpips = validate_epoch(model, val_loader, epoch, perception_loss_fn=lpips.LPIPS(net='vgg').to(configs.device))
        epoch_list.append(epoch)
        val_loss_list.append(val_loss)
        val_psnr_list.append(val_psnr)
        val_ssim_list.append(val_ssim)
        val_lpips_list.append(val_lpips_list)

    eva_logs = pd.DataFrame({'epoch': epoch_list,
                             'val_loss': val_loss_list,
                             'val_psnr': val_psnr_list,
                             'val_ssim': val_ssim_list,
                             'val_lpips': val_lpips_list
                             })
    eva_logs.to_csv('./logs/{}.csv'.format(savename))
