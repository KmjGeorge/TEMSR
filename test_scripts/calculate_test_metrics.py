import os

import matplotlib.pyplot as plt
import numpy as np
import pyiqa
import skimage.metrics
import torchvision.io
from PIL import Image
from torchvision.io import ImageReadMode
import torch
import torch.nn.functional as F
from basicsr.metrics.psnr_ssim import calculate_ssim, calculate_psnr
from basicsr.metrics.niqe import calculate_niqe
from tqdm import tqdm


# img_path = 'D:/Datasets/TEMPatch for SR/1/Original/1.png'
# img = Image.open(img_path).convert('L')
# bic = img.resize((512, 512), Image.BICUBIC)
# bic.save(img_path.replace('1.png', '1_bic2x.png'))

# niqe = pyiqa.create_metric('niqe')
# img_bic = (torchvision.io.read_image('D:\Datasets\TEMPatch for SR\\1\Original\\1_bic2x.png') / 255.0).unsqueeze(0)
# img_pred = (torchvision.io.read_image('D:\Datasets\TEMPatch for SR\\1\Output\\1_RCAN_Internal.png', mode=ImageReadMode.GRAY) / 255.0).unsqueeze(0)
# print(img_bic.shape, img_pred.shape)
# niqe1 = niqe(img_bic).item()
# niqe2 = niqe(img_pred).item()
# print(niqe1, niqe2)


def get_bic(path, save_path, resize_shape=(256, 256)):
    for filename in os.listdir(path):
        img = Image.open(os.path.join(path, filename)).convert('L')
        img_bic = img.resize(resize_shape, Image.BICUBIC)
        img_bic.save(os.path.join(save_path, filename))


# def cal_bic_metrics(folder_bic, folder_gt):
#     img_bic = []
#     img_gt = []
#     psnr = 0
#     ssim = 0
#     niqe = 0
#     for filename in os.listdir(folder_bic):
#         # if '_bic2x' in filename:
#         img_bic.append(np.array(Image.open(os.path.join(folder_bic, filename)).convert('L')))
#
#     for filename in os.listdir(folder_gt):
#         img_gt.append(np.array(Image.open(os.path.join(folder_gt, filename)).convert('L')))
#
#     for im1, im2 in zip(img_gt, img_bic):
#         ssim += calculate_ssim(im1, im2, crop_border=0)
#         psnr += calculate_psnr(im1, im2, crop_border=0)
#         niqe += calculate_niqe(im2, crop_border=0)
#     return psnr / len(img_bic), ssim / len(img_bic), niqe / len(img_bic)


def cal_test_metrics(folder_pred, folder_gt):
    img_pred = []
    img_gt = []
    psnr_avg = 0
    ssim_avg = 0
    niqe_avg = 0
    pred_list = os.listdir(folder_pred)
    pred_list.sort(key=lambda x: int(x.split('.')[0][0:5]))
    gt_list = os.listdir(folder_gt)
    gt_list.sort(key=lambda x: int(x.split('.')[0][0:5]))
    for filename in pred_list:
        img_pred.append(np.array(Image.open(os.path.join(folder_pred, filename)).convert('L')))

    for filename in gt_list:
        img_gt.append(np.array(Image.open(os.path.join(folder_gt, filename)).convert('L')))

    loop = tqdm(zip(img_pred, img_gt))
    for im1, im2 in loop:
        psnr = calculate_psnr(im1, im2, crop_border=0)
        ssim = calculate_ssim(im1, im2, crop_border=0)
        niqe = calculate_niqe(im1, crop_border=0)
        psnr_avg += psnr
        ssim_avg += ssim
        niqe_avg += niqe
        # plt.subplot(121)
        # plt.imshow(im1, 'gray')
        # plt.subplot(122)
        # plt.imshow(im2, 'gray')
        # plt.show()
        loop.set_postfix({'psnr':psnr, 'ssim':ssim, 'niqe':niqe})
    return psnr_avg / len(img_pred), ssim_avg / len(img_pred), niqe_avg / len(img_pred)


if __name__ == '__main__':
    print(cal_test_metrics(
        folder_pred='D:\\github\\TEMSR_BasicSR\\results\\111_RCANx4_scratch_TEMImageNet_bat8_inc3feature64group6block10_lr1e-4_test(ImageNet)\\visualization\\TEMImageNet',
        folder_gt='D:\Datasets\TEM-ImageNet-v1.3-master\\noBackgroundnoNoise\\full'))
    # print(cal_test_metrics(
    #     folder_pred='D:\github\TEMSR_BasicSR\\results\\101_RCANx2_scratch_TEMImageNet_bat8_inc3feature64group6block10_lr1e-4_test\\visualization\TEMSR\\val',
    #     folder_gt='D:\Datasets\TEMPatch for SR\GT\Val'))
