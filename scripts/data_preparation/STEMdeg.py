import math
import os
import random

import matplotlib.pyplot as plt
import cv2
import numpy as np
import tifffile as tiff
import torchvision.io
from torch import Tensor
from torchvision.io import ImageReadMode

from basicsr.archs.vgg_arch import VGGFeatureExtractor
import torch.nn.functional as F
import torch
from PIL import Image

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.losses.cxloss import symetric_CX_loss
from basicsr.utils.img_process_util import filter2D
from tqdm import tqdm


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


setup_seed(12345)


def adaptive_threshold(img_tensor, neighborhood_size=15, threshold_factor=1.5):
    """
    对图像应用自适应阈值，使得亮区域变暗，暗区域几乎不变。

    参数:
    - image: 输入的图像张量，形状为 (C, H, W) 或 (1, H, W)。
    - k: 用于调整局部阈值的系数。
    - block_size: 局部区域的大小，用于计算局部阈值。

    返回:
    - 调整后的图像张量。
    """
    # 计算局部平均值和标准差
    # 计算局部均值
    # 计算局部均值
    local_mean = F.avg_pool2d(img_tensor, kernel_size=neighborhood_size, stride=1,
                              padding=(neighborhood_size - 1) // 2)
    result = torch.clamp(img_tensor - torch.abs(local_mean - img_tensor) * threshold_factor, 0, 1)

    return result


def luminance_threshold(img_tensor, threshold=0.8):
    max = img_tensor.max()
    result = torch.clamp(img_tensor, 0, max * threshold)
    return result


def adjust_contrast(img_tensor, contrast_factor):
    """

    Args:
        img_tensor: (C, H, W)
        contrast_facotr: contrast adjust factor
    Returns:
        The adjusted image tensor\
    """
    mean = img_tensor.mean()  # 计算图像的平均亮度
    # 调整对比度
    contrasted_image = (img_tensor - mean) * contrast_factor + mean
    # 确保数值范围不会超出[0, 1]
    contrasted_image = contrasted_image.clip(0, 1)

    return contrasted_image


def linear_exposure_compensation(image_array, maxval, compensation_value):
    # 确保补偿值在合理的范围内
    compensation_value = max(-1, min(1, compensation_value))

    # 应用曝光补偿
    # 这里使用线性变换来调整曝光
    # 公式为: new_value = scale * (original_value - mid_value) + mid_value
    # 其中 scale 是补偿因子，mid_value 是像素值范围的中点（例如，对于 uint8 范围是 128）
    mid_value = maxval / 2  # 对于 uint8 类型的图像，中点是 128
    scale = 1 + compensation_value  # 计算缩放因子

    # 对每个颜色通道进行曝光补偿
    adjusted_image = scale * (image_array - mid_value) + mid_value

    # 确保调整后的像素值在范围内
    adjusted_image = torch.clip(adjusted_image, 0, maxval)

    return adjusted_image


def rgb_to_grayscale(img: Tensor, num_output_channels: int = 1) -> Tensor:
    if num_output_channels not in (1, 3):
        raise ValueError("num_output_channels should be either 1 or 3")

    if img.shape[-3] == 3:
        r, g, b = img.unbind(dim=-3)
        # This implementation closely follows the TF one:
        # https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L2105-L2138
        l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
        l_img = l_img.unsqueeze(dim=-3)
    else:
        l_img = img.clone()

    if num_output_channels == 3:
        return l_img.expand(img.shape)

    return l_img


def generate_poisson_noise_pt(img, scale=1.0, gray_noise=0):
    """Generate a batch of poisson noise (PyTorch version)

    Args:
        img (Tensor): Input image, shape (b, c, h, w), range [0, 1], float32.
        scale (float | Tensor): Noise scale. Number or Tensor with shape (b).
            Default: 1.0.
        gray_noise (float | Tensor): 0-1 number or Tensor with shape (b).
            0 for False, 1 for True. Default: 0.

    Returns:
        (Tensor): Returned noisy image, shape (b, c, h, w), range[0, 1],
            float32.
    """
    b, _, h, w = img.size()
    if isinstance(gray_noise, (float, int)):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0
    if cal_gray_noise:
        img_gray = rgb_to_grayscale(img, num_output_channels=1)
        # round and clip image for counting vals correctly
        img_gray = torch.clamp((img_gray * 255.0).round(), 0, 255) / 255.
        # use for-loop to get the unique values for each sample
        vals_list = [len(torch.unique(img_gray[i, :, :, :])) for i in range(b)]
        vals_list = [2 ** np.ceil(np.log2(vals)) for vals in vals_list]
        vals = img_gray.new_tensor(vals_list).view(b, 1, 1, 1)
        out = torch.poisson(img_gray * vals) / vals
        noise_gray = out - img_gray
        noise_gray = noise_gray.expand(b, 3, h, w)

    # always calculate color noise
    # round and clip image for counting vals correctly
    img = torch.clamp((img * 255.0).round(), 0, 255) / 255.
    # use for-loop to get the unique values for each sample
    vals_list = [len(torch.unique(img[i, :, :, :])) for i in range(b)]
    vals_list = [2 ** np.ceil(np.log2(vals)) for vals in vals_list]
    vals = img.new_tensor(vals_list).view(b, 1, 1, 1)
    out = torch.poisson(img * vals) / vals
    noise = out - img
    if cal_gray_noise:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    if not isinstance(scale, (float, int)):
        scale = scale.view(b, 1, 1, 1)
    return noise * scale


def random_generate_poisson_noise_pt(img, scale_range=(0, 1.0), gray_prob=0):
    scale = torch.rand(
        img.size(0), dtype=img.dtype, device=img.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
    gray_noise = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
    gray_noise = (gray_noise < gray_prob).float()
    return generate_poisson_noise_pt(img, scale, gray_noise)


def generate_gaussian_noise_pt(img, sigma=10, gray_noise=0):
    """Add Gaussian noise (PyTorch version).

    Args:
        img (Tensor): Shape (b, c, h, w), range[0, 1], float32.
        scale (float | Tensor): Noise scale. Default: 1.0.

    Returns:
        (Tensor): Returned noisy image, shape (b, c, h, w), range[0, 1],
            float32.
    """
    b, _, h, w = img.size()
    if not isinstance(sigma, (float, int)):
        sigma = sigma.view(img.size(0), 1, 1, 1)
    if isinstance(gray_noise, (float, int)):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0

    if cal_gray_noise:
        noise_gray = torch.randn(*img.size()[2:4], dtype=img.dtype, device=img.device) * sigma / 255.
        noise_gray = noise_gray.view(b, 1, h, w)

    # always calculate color noise
    noise = torch.randn(*img.size(), dtype=img.dtype, device=img.device) * sigma / 255.

    if cal_gray_noise:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    return noise


def random_add_poisson_noise_pt(img, scale_range=(0, 1.0), gray_prob=0, clip=True, rounds=False):
    noise = random_generate_poisson_noise_pt(img, scale_range, gray_prob)
    out = img + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out, noise


def random_generate_gaussian_noise_pt(img, sigma_range=(0, 10), gray_prob=0):
    sigma = torch.rand(
        img.size(0), dtype=img.dtype, device=img.device) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
    gray_noise = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
    gray_noise = (gray_noise < gray_prob).float()
    return generate_gaussian_noise_pt(img, sigma, gray_noise)


def random_add_gaussian_noise_pt(img, sigma_range=(0, 1.0), gray_prob=0, clip=True, rounds=False, scale_range=[1.0, 1.0]):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    noise = random_generate_gaussian_noise_pt(img, sigma_range, gray_prob)
    noise = noise * scale
    out = img + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out, noise


def tensor2inp(img_tensor):
    img_tensor = img_tensor.squeeze()
    img_np = img_tensor.detach().cpu().numpy()
    img_np *= 255
    img_np = img_np.astype(np.uint8)
    return img_np


def add_heteroscedastic_gnoise(image, device, sigma_1_range=(5e-3, 5e-2), sigma_2_range=(1e-3, 1e-2), scale=1.0):
    """
    Adds heteroscedastic Gaussian noise to an image.

    Parameters:
    - image: PyTorch tensor of the image.
    - sigma_1_range: Tuple indicating the range of sigma_1 values.
    - sigma_2_range: Tuple indicating the range of sigma_2 values.

    Returns:
    - Noisy image: Image tensor with added heteroscedastic Gaussian noise.
    """
    # Randomly choose sigma_1 and sigma_2 within the specified ranges
    sigma_1 = torch.empty(image.size()).uniform_(*sigma_1_range).to(device)
    sigma_2 = torch.empty(image.size()).uniform_(*sigma_2_range).to(device)

    # Calculate the variance for each pixel
    variance = (sigma_1 ** 2) * image + (sigma_2 ** 2)
    # if variance.le(0):
    #     print('aaaaa'),
    #     print(torch.where(image < 0))
    #     variance = 0
    # Generate the Gaussian noise
    std = variance.sqrt()
    noise = torch.normal(mean=0.0, std=std)

    # Add the noise to the original image
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0., 1.), noise


def deg(gt_img, noise='poisson-gaussian',
        compensation_range=[0., 0.],
        neighborhood_size=15,
        threshold_factor_range=[0.4, 0.6],
        contrast_range=[1., 1.],
        scale_range=[0., 1.],
        sigma_range=[0., 1.],
        resize_prob=[0., 0., 1],
        resize_range=[0.3, 1.5],
        sigma_1_range=[5e-3, 5e-2],
        sigma_2_range=[1e-3, 1e-2],
        randomshuffle=False):
    gt_img = gt_img / 255.0
    out = gt_img
    # out = adaptive_threshold(gt_img, neighborhood_size, threshold_factor)
    # gt_noisy, noise = add_natural_noise(gt_img_tensor, use_cuda=False)
    if randomshuffle:
        seq = np.random.choice([1, 2, 3, 4], size=4, replace=False)
    else:
        seq = [1, 2, 3, 4]
    for step in seq:
        if step == 1:
            '''
            threshold_factor = np.random.uniform(threshold_factor_range[0], threshold_factor_range[1])
            out = adaptive_threshold(out, neighborhood_size=neighborhood_size, threshold_factor=threshold_factor)
            compensation_value = np.random.uniform(compensation_range[0], compensation_range[1])
            out = linear_exposure_compensation(out, maxval=1, compensation_value=compensation_value)
            '''
            contrast_factor = np.random.uniform(contrast_range[0], contrast_range[1])
            out = adjust_contrast(out, contrast_factor=contrast_factor)
        elif step == 2:
            compensation_factor = np.random.uniform(compensation_range[0], compensation_range[1])
            out = linear_exposure_compensation(out, 1, compensation_factor)
        elif step == 3:
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], resize_prob)[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, resize_range[1])
            elif updown_type == 'down':
                scale = np.random.uniform(resize_range[0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
        elif step == 4:
            if noise == 'poisson-gaussian':
                out, noise1 = random_add_poisson_noise_pt(out, scale_range=scale_range, gray_prob=1)  # poisson noise
                out, noise2 = random_add_gaussian_noise_pt(
                    out, sigma_range=sigma_range, gray_prob=1, scale=5)
            elif noise == 'heteroscedastic_gnoise':
                out, noise = add_heteroscedastic_gnoise(out, device='cpu', sigma_1_range=sigma_1_range,
                                                        sigma_2_range=sigma_2_range)
            elif noise == 'nonoise':
                noise = None
            else:
                raise 'Error noise!'
    out = torch.clamp((out[:, 0, :, :].unsqueeze(1) * 255.0).round(), 0, 255) / 255.

    return out, noise


def generate_kernel(kernel_range=[2 * v + 1 for v in range(3, 7)],
                    sinc_prob=0.,
                    kernel_list=['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
                    kernel_prob=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
                    blur_sigma=[0.1, 1],
                    betag_range=[0.1, 0.5],
                    betap_range=[0.1, 0.5]):
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < sinc_prob:
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel = random_mixed_kernels(
            kernel_list,
            kernel_prob,
            kernel_size,
            blur_sigma,
            blur_sigma, [-math.pi, math.pi],
            betag_range,
            betap_range,
            noise_range=None)
    # pad kernel
    pad_size = (21 - kernel_size) // 2
    kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    kernel = torch.FloatTensor(kernel)
    return kernel


def deg_from_sim(gt_img, noise='poisson-gaussian',
                 use_blur=False,
                 compensation_range=[0., 0.],
                 contrast_range=[1., 1.],
                 scale_p_range=[1., 1.],
                 sigma_range=[1., 1.],
                 scale_g_range=[1., 1.],
                 resize_prob=[0., 0., 1],
                 resize_range=[0.3, 1.5],
                 sigma_1_range=[5e-3, 5e-2],
                 sigma_2_range=[1e-3, 1e-2],
                 sigma_jitter_range=[2, 5],
                 scan_noise_prob=0.2,
                 randomshuffle=False):
    gt_img = gt_img / 255.0
    out = gt_img
    orig_h, orig_w = gt_img.size()[2:4]
    # out = adaptive_threshold(gt_img, neighborhood_size, threshold_factor)
    # gt_noisy, noise = add_natural_noise(gt_img_tensor, use_cuda=False)
    if randomshuffle:
        if use_blur:
            seq = np.random.choice([1, 2, 3, 4, 5, 6], size=6, replace=False)
        else:
            seq = np.random.choice([2, 3, 4, 5, 6], size=5, replace=False)
    else:
        if use_blur:
            seq = [1, 2, 3, 4, 5, 6]
        else:
            seq = [2, 3, 4, 5, 6]
    seq = list(seq)
    for step in seq:
        if step == 1:
            kernel = generate_kernel()
            out = filter2D(out, kernel)
        if step == 2:
            contrast_factor = np.random.uniform(contrast_range[0], contrast_range[1])
            out = adjust_contrast(out, contrast_factor=contrast_factor)
        elif step == 3:
            compensation_factor = np.random.uniform(compensation_range[0], compensation_range[1])
            out = linear_exposure_compensation(out, 1, compensation_factor)
        elif step == 4:
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], resize_prob)[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, resize_range[1])
            elif updown_type == 'down':
                scale = np.random.uniform(resize_range[0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # resize back
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(orig_h, orig_w), mode=mode)
        elif step == 5:
            if noise == 'poisson-gaussian':
                out, noise1 = random_add_poisson_noise_pt(out, scale_range=scale_p_range, gray_prob=1)  # poisson noise
                out, noise2 = random_add_gaussian_noise_pt(
                    out, sigma_range=sigma_range, gray_prob=1, scale_range=scale_g_range)
                out = out[:, 0, :, :].unsqueeze(0)
            elif noise == 'heteroscedastic_gnoise':
                out, noise = add_heteroscedastic_gnoise(out, device='cpu', sigma_1_range=sigma_1_range,
                                                        sigma_2_range=sigma_2_range)
            elif noise == 'nonoise':
                noise = None
            else:
                raise 'Error noise!'
        elif step == 6:
            if np.random.uniform() < scan_noise_prob:
                sigma_jitter = np.random.uniform(sigma_jitter_range[0], sigma_jitter_range[1])
                out = add_scan_noise(out, sigma_jitter, phi=np.pi / 2)

    out = torch.clamp((out * 255.0).round(), 0, 255) / 255.

    return out, noise, seq


def grayimg2tensor(gray):
    gray = torch.from_numpy(gray)
    gray = gray.unsqueeze(0).unsqueeze(0)
    return gray


def show(lq_img, gt_img, noise, gt_deg):
    plt.subplot(221)
    plt.title('GT')
    plt.imshow(gt_img, 'gray')
    plt.subplot(222)
    plt.title('LQ')
    plt.imshow(lq_img, 'gray')
    plt.subplot(223)
    plt.title('Noise')
    plt.imshow(noise, 'gray')
    plt.subplot(224)
    plt.title('Deg GT')
    plt.imshow(gt_deg, 'gray')
    plt.tight_layout()
    plt.show()


def vgg_feature(img, layer='conv5_4'):
    layer_weights = {layer: 1}
    vgg = VGGFeatureExtractor(
        layer_name_list=list(layer_weights.keys()),
        vgg_type='vgg19',
        use_input_norm=True,
        range_norm=False)
    img_feature = vgg(img)[layer]
    return img_feature


def caculate_cxloss(vgg, img1, img2, layer='conv5_4'):
    gt_feature = vgg(img1)[layer]
    lq_feature = vgg(img2)[layer]
    cx_loss = symetric_CX_loss(gt_feature, lq_feature)
    return cx_loss.item()


def calucate_cxloss_folder(folder1, folder2, layer='conv5_4'):
    layer_weights = {layer: 1}
    vgg = VGGFeatureExtractor(
        layer_name_list=list(layer_weights.keys()),
        vgg_type='vgg19',
        use_input_norm=True,
        range_norm=False)
    cx_loss = 0
    length = len(os.listdir(folder1))
    for file1, file2 in zip(os.listdir(folder1), os.listdir(folder2)):
        img1 = tiff.imread(os.path.join(folder1, file1))
        img2 = tiff.imread(os.path.join(folder2, file2))
        feature1 = vgg(img1)[layer]
        feature2 = vgg(img2)[layer]
        cx_loss += symetric_CX_loss(feature1, feature2)
    cx_loss /= length
    return cx_loss.item()


def generate_degval(gt_path, save_lq_path):
    from tqdm import tqdm

    for filename in tqdm(os.listdir(gt_path)):
        gt_img = cv2.imread(os.path.join(gt_path, filename), 0)
        gt_tensor = grayimg2tensor(gt_img)
        deg_gt, noise = deg(gt_tensor, noise='poisson-gaussian',
                            contrast_range=[1., 1.],
                            compensation_range=[0.0, 0.0],
                            scale_range=[10, 10],
                            sigma_range=[10, 10],
                            randomshuffle=False)

        deg_gt_img = tensor2inp(deg_gt)
        cv2.imwrite(os.path.join(save_lq_path, filename), deg_gt_img)
    print('Done!')


def add_scan_noise(img, sigma_jitter=0.2, phi=np.pi / 4, f=1 / 200):
    h, w = img.shape[2], img.shape[3]
    img_pix = img.squeeze()
    img_new = torch.zeros(size=(h, w))
    for i in range(h):
        delta_x = int((np.random.normal() * sigma_jitter * np.sin(2 * np.pi * f * i)).round())
        for j in range(w):
            delta_y = int((np.random.normal() * sigma_jitter * np.sin(2 * np.pi * f * j + phi)).round())
            try:
                img_new[i][j] = img_pix[i - delta_x][j - delta_y]
            except:
                img_new[i][j] = img_pix[i][j]
    img_new = img_new.unsqueeze(0).unsqueeze(0)
    return img_new


def make_deg_folder(n_thread=10, orig_folder='F:\Datasets\Sim ReSe2\\all_crops',
                    save_gt_folder='F:\Datasets\Sim ReSe2\\all_crops_GT',
                    save_lq_folder='F:\Datasets\Sim ReSe2\\all_crops_LQ',
                    repeats=20):
    from multiprocessing import Pool
    from tqdm import tqdm

    img_list = os.listdir(orig_folder)
    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(n_thread)
    for filename in img_list:
        orig_path = os.path.join(orig_folder, filename)
        if save_gt_folder:
            save_gt_path = os.path.join(save_gt_folder, filename)
        else:
            save_gt_path = None
        save_lq_path = os.path.join(save_lq_folder, filename)
        pool.apply_async(worker, args=(orig_path, save_gt_path, save_lq_path, repeats),
                         callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def worker(orig_path, save_gt_path, save_lq_path, repeats):
    img_sim = cv2.imread(orig_path, 0)
    img_sim_pt = grayimg2tensor(img_sim)
    for it in range(repeats):
        if save_gt_path:
            cv2.imwrite(save_gt_path.replace('.png', '_{}.png'.format(it)), img_sim)
        img_deg_pt, _, _ = deg_from_sim(img_sim_pt,
                                        use_blur=False,
                                        noise='poisson-gaussian',
                                        contrast_range=[0.3, 0.5],
                                        compensation_range=[0.2, 0.35],
                                        resize_prob=[0.2, 0.7, 0.1],
                                        resize_range=[0.25, 2],
                                        scale_p_range=[3.0, 3.0],
                                        sigma_range=[3.0, 3.0],
                                        scale_g_range=[5, 10],
                                        scan_noise_prob=1.0,
                                        sigma_jitter_range=[2, 5],
                                        randomshuffle=False)
        img_deg = tensor2inp(img_deg_pt)
        cv2.imwrite(save_lq_path.replace('.png', '_{}.png'.format(it)), img_deg)


if __name__ == '__main__':
    make_deg_folder(n_thread=8, orig_folder='D:\Datasets\Sim ReSe2\yolo\\roboflow\simGT\\val\images',
                    save_gt_folder=None,
                    save_lq_folder='D:\Datasets\Sim ReSe2\yolo\\roboflow\simLQ\\val\images', repeats=1)

    '''
    # folder deg
    gt_folder = 'F:\Datasets\Sim ReSe2\\all_crops'
    lq_folder = 'F:\Datasets\Sim ReSe2\\simlq'
    for filename in tqdm(os.listdir(gt_folder)):
        img_sim = cv2.imread(os.path.join(gt_folder, filename), 0)
        img_sim_pt = grayimg2tensor(img_sim)
        img_deg_pt, _ = deg_from_sim(img_sim_pt,
                                     use_blur=False,
                                     compensation_range=[0.3, 0.6],
                                     contrast_range=[0.4, 0.6],
                                     resize_prob=[0., 0., 1.],
                                     resize_range=[0.3, 1.5],
                                     scale_range=[4, 8],
                                     sigma_range=[1, 5],
                                     scan_noise_prob=0.,
                                     sigma_jitter=5)
        img_deg = tensor2inp(img_deg_pt)
        cv2.imwrite(os.path.join(lq_folder, filename), img_deg)
    '''

    # plt.imshow(img_deg, 'gray')
    # plt.savefig()
    # plt.show()
    #

    '''
    gt_img = cv2.imread('D:\Datasets\Sim ReSe2\\all\\0.3_0.0_0_0_35.0_0.6_1728.png', 0).astype(np.float32)
    lq_img = cv2.imread('D:\Datasets\STEM ReSe2\ReSe2\singlelayer\LQ\crops1024\\20230829 2015 7.70 Mx 13 nm HAADF_s001.png', 0).astype(np.float32)
    gt_tensor, lq_tensor = grayimg2tensor(gt_img).cuda(), grayimg2tensor(lq_img).cuda()
    lq_tensor /= 255.0
    # hyper search
    layer_weights = {'conv5_4': 1}
    vgg = VGGFeatureExtractor(
        layer_name_list=list(layer_weights.keys()),
        vgg_type='vgg19',
        use_input_norm=True,
        range_norm=False).cuda()
    compensation_values = np.arange(-0.4, 0.4, 0.1)
    contrast_factors = np.arange(0.1, 1.0, 0.1)
    scale_ranges = np.arange(0.5, 5, 0.5)
    sigma_ranges = np.arange(0.5, 5, 0.5)
    max_iters = 10000
    best_cx = 99999999
    best_mean = 99999999
    best_std = 99999999
    with open('D:\github\TEMSR\experiments\\hyper_log(sim).txt', 'w') as f:
        for iter in tqdm(range(max_iters)):
            compensation_value = np.random.choice(compensation_values).round(2)
            contrast_factor = np.random.choice(contrast_factors).round(2)
            scale_range = np.random.choice(scale_ranges).round(2)
            sigma_range = np.random.choice(sigma_ranges).round(2)
            deg_gt, noise, shuffle = deg_from_sim(gt_tensor,
                                                  noise='poisson-gaussian',
                                                  use_blur=False,
                                                  contrast_range=[contrast_factor, contrast_factor],
                                                  compensation_range=[compensation_value, compensation_value],
                                                  scale_range=[scale_range, scale_range],
                                                  sigma_range=[sigma_range, sigma_range],
                                                  scan_noise_prob=0.,
                                                  randomshuffle=True)
            shuffle.remove(4)
            deg_gt_img = tensor2inp(deg_gt)

            cx_loss = caculate_cxloss(vgg, deg_gt, lq_tensor)
            mean_dis = math.fabs(lq_img.mean() - deg_gt_img.mean())
            std_dis = math.fabs(lq_img.std() - deg_gt_img.std())
            f.write(str(shuffle) + ' : ' + str(contrast_factor) + ' ' + str(compensation_value) + ' ' + str(
                scale_range) + ' ' + str(
                sigma_range) + ' ====> ')
            f.write('cx_loss = ' + str(cx_loss) + ' ' + 'mean_dis = ' + str(mean_dis) + ' ' + 'std_dis = ' + str(
                std_dis) + ' ')
            # print(compensation_value, contrast_factor, scale_range, sigma_range, end=' ====> ')
            # print('cx_loss =', cx_loss, end=' ')
            # print('mean_dis =', mean_dis, end=' ')
            # print('std_dis =', std_dis, end=' ')
            if cx_loss < best_cx:
                best_cx = cx_loss
                best_param_cx = [shuffle, contrast_factor, compensation_value, scale_range, sigma_range]
                f.write('Best CX! ')
                # print('Best CX!', end=' ')
            if mean_dis < best_mean:
                best_mean = mean_dis
                best_param_mean = [shuffle, contrast_factor, compensation_value, scale_range, sigma_range]
                f.write('Best Mean! ')
                # print('Best Mean!', end=' ')
            if std_dis < best_std:
                best_std = std_dis
                best_param_std = [shuffle, contrast_factor, compensation_value, scale_range, sigma_range]
                f.write('Best Std! ')
                # print('Best Std!', end=' ')
            # print(' ')
            f.write('\n')
        f.write('Best Mean, Best Std, Best CX  =======> ' + str(best_mean) + ' ' + str(best_std) + ' ' + str(best_cx))
        f.write(
            'Best Params for Mean Std Cx  =======>  ' + str(best_param_mean) + ' ' + str(best_param_std) + ' ' + str(
                best_param_cx))
        print('Best Mean, Best Std, Best CX', end='  =======> ')
        print(best_mean, best_std, best_cx)
        print('Best Params for Mean Std Cx', end='  =======>  ')
        print(best_param_mean, best_param_std, best_param_cx)

    '''
    '''
    from tqdm import tqdm

    layer_weights = {'conv5_4': 1}
    vgg = VGGFeatureExtractor(
        layer_name_list=list(layer_weights.keys()),
        vgg_type='vgg19',
        use_input_norm=True,
        range_norm=False)
    lq_path = 'G:\datasets\STEM ReSe2\ReSe2\\all_LQ'
    lq_files = os.listdir(lq_path)
    gt_path = 'G:\datasets\STEM ReSe2\ReSe2\\all_GT'
    gt_files = os.listdir(gt_path)
    noise_scale = 10
    loss = 0
    pairs = 0
    deg_mean, deg_var = 0, 0
    for gt_file in tqdm(gt_files):
        if gt_file.replace('LQ', 'GT') in lq_files:
            pairs += 1
            lq_file = gt_file.replace('LQ', 'GT')
            lq_img = tiffile.imread(os.path.join(lq_path, lq_file))
            gt_img = tiffile.imread(os.path.join(gt_path, gt_file))
            lq_img, gt_img = cv2.cvtColor(lq_img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
            lq_tensor = grayimg2tensor(lq_img)
            gt_tensor = grayimg2tensor(gt_img)
            gt_deg_tensor, noise_tensor = deg(gt_tensor, noise='poisson', scale_range=[noise_scale, noise_scale])
            loss += caculate_cxloss(vgg, gt_deg_tensor, lq_tensor)
        else:
            gt_img = tiffile.imread(os.path.join(gt_path, gt_file))
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
            gt_tensor = grayimg2tensor(gt_img)
            gt_deg_tensor, noise_tensor = deg(gt_tensor, noise='poisson', scale_range=[noise_scale, noise_scale])
        deg_mean += gt_deg_tensor.mean().item()
        deg_var += gt_deg_tensor.var().item()
        gt_deg = tensor2inp(gt_deg_tensor)
        gt_deg_PIL = Image.fromarray(gt_deg)
        gt_deg_PIL.save(
            'G:\datasets\STEM ReSe2\ReSe2\\all_deg({})\\{}'.format(noise_scale, gt_file.replace('.tif', '.png')))
        # show(lq_img, gt_img, noise, gt_deg)

    loss /= pairs
    deg_mean /= len(gt_files)
    deg_var /= len(gt_files)
    deg_std = np.sqrt(deg_var)
    print(loss, deg_mean, deg_std)
    '''
