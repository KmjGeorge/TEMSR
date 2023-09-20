import torch

from dataset.temimagenet import get_temimagenet_trainval
from models.SwinIR import get_swinir
from models.UHDFour import get_uhdfour
from train.train import validate_epoch
import configs
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import cal_ssim, cal_psnr
from utils.functions import image_for_network, image_for_draw
if __name__ == '__main__':

    savename = 'uhdfour_3loss'
    model = get_uhdfour().to(configs.device)
    model.load_state_dict(torch.load('./weights/{}_epoch30.pt'.format(savename)))
    model.eval()
    filename = '00773'
    pic = Image.open('D:/Datasets/TEM-ImageNet-v1.3-master/image/{}.png'.format(filename)).resize(
        configs.multiscale_aug_config['orig_size'], Image.Resampling.LANCZOS).convert(
        'L')
    gt = Image.open('D:/Datasets/TEM-ImageNet-v1.3-master/noNoise/{}.png'.format(filename)).resize(
        configs.multiscale_aug_config['orig_size'], Image.Resampling.LANCZOS).convert('L')
    gt_tensor = image_for_network(gt, target_c=3)
    pic_tensor = image_for_network(pic, target_c=3)
    output = model(pic_tensor)

    out_pic = image_for_draw(output)
    plt.subplot(131)
    plt.imshow(pic, 'gray')
    plt.title('Original')
    plt.subplot(132)
    plt.imshow(out_pic, 'gray')
    plt.title('Output')
    plt.subplot(133)
    plt.imshow(gt, 'gray')
    plt.title('GT')
    plt.tight_layout()
    plt.show()

    print('psnr=', cal_psnr(output, gt_tensor).item())
    print('ssim=', cal_ssim(output, gt_tensor).item())
