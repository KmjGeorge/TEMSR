import math
import os
import random
import shutil

import matplotlib.pyplot as plt
import cv2
import numpy as np
import numpy.random
import tiffile as tiff
import torchvision.io
from torch import Tensor
from torchvision.io import ImageReadMode

from basicsr.archs.vgg_arch import VGGFeatureExtractor
import torch.nn.functional as F
import torch
from PIL import Image
from torchvision.transforms import RandomRotation
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.losses.cxloss import symetric_CX_loss
from basicsr.utils.img_process_util import filter2D
from tqdm import tqdm

seed = 12345


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_row_line_noise(img, row_factor_range=[0.5, 1.5]):
    h, w = img.shape
    noisy_img = img.copy()
    for i in range(np.random.randint(1, 3)):
        width = np.random.randint(1, 5)
        row_idx = np.random.randint(0, h - width)
        length = np.random.randint(w // 2, w)
        row = img[row_idx:row_idx + width,
              int((w - length) * 0.5):int((w + length) * 0.5)]  # (width, length)
        row_mean = np.mean(row, axis=1, keepdims=True)  # (width, 1)
        factors = np.random.uniform(row_factor_range[0], row_factor_range[1], size=(width, length))
        row = row_mean * factors
        noisy_img[row_idx: row_idx + width, int((w - length) * 0.5):int((w + length) * 0.5)] = row
    return noisy_img


def add_black_pixel_noise(img, num_points):
    if num_points == 0:
        return img
    h, w = img.shape
    noisy_img = img.copy()
    for i in range(np.random.randint(1, num_points)):
        bh, bw = np.random.randint(0, h), np.random.randint(0, w)
        noisy_img[bh, bw] = 0.
    return noisy_img


def add_zinger_pixel_noise(img, num_points, maxval=1.):
    if num_points == 0:
        return img
    h, w = img.shape
    noisy_img = img.copy()
    for i in range(np.random.randint(1, num_points)):
        bh, bw = np.random.randint(0, h), np.random.randint(0, w)
        noisy_img[bh, bw] = maxval
    return noisy_img


def adjust_contrast(img, contrast_factor, maxval):
    mean = img.mean()
    adjusted_image = (img - mean) * contrast_factor + mean
    adjusted_image = adjusted_image.clip(0, maxval)
    return adjusted_image


def adjust_brightness(img, bright_factor, maxval):
    adjusted_image = np.clip(bright_factor * img, 0, maxval)
    return adjusted_image


def tensor2inp(img_tensor):
    img_tensor = img_tensor.squeeze()
    img_np = img_tensor.detach().cpu().numpy()
    img_np *= 255
    img_np = img_np.astype(np.uint8)
    return img_np

def gaussian_2d(shape=(512, 512), center=None):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    if center is None:
        center = (0, 0)
    x0, y0 = center
    sigma_x, sigma_y = shape[1] // 2, shape[0] // 6
    g = np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)))
    g_max, g_min = g.max(), g.min()
    g = (g - g_min) / (g_max - g_min)
    return g



def generate_kernel(kernel_range=[2 * v + 1 for v in range(3, 7)],
                    sinc_prob=0.,
                    kernel_list=['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso',
                                 'plateau_aniso'],
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


def add_poisson_noise(image, lamb_range, noise=None):
    if noise is None:
        lamb = np.random.uniform(lamb_range[0], lamb_range[1])
        noisy_image = np.float32(np.random.poisson(image * lamb) / lamb)
        noisy_image_max, noisy_image_min = noisy_image.max(), noisy_image.min()
        noisy_image = (noisy_image - noisy_image_min) / (noisy_image_max - noisy_image_min)
        noise = noisy_image - image
    noisy_image = image + noise
    return noisy_image, noise


def add_poisson_noise_basicsr(img, scale_range, clip=True, rounds=False):
    def generate_poisson_noise(img, scale=1.0):
        """Generate poisson noise.

        Reference: https://github.com/scikit-image/scikit-image/blob/main/skimage/util/noise.py#L37-L219

        Args:
            img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
            scale (float): Noise scale. Default: 1.0.

        Returns:
            (Numpy array): Returned noisy image, shape (h, w, c), range[0, 1],
                float32.
        """
        # round and clip image for counting vals correctly
        img = np.clip((img * 255.0).round(), 0, 255) / 255.
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        out = np.float32(np.random.poisson(img * vals) / float(vals))
        noise = out - img
        return noise * scale

    def random_generate_poisson_noise(img, scale_range=(0, 1.0)):
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return generate_poisson_noise(img, scale)

    noise = random_generate_poisson_noise(img, scale_range)
    out = img + noise

    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out, noise


def add_gaussian_noise(image, sigma_range, noise=None):
    if noise is None:
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])
        noise = np.random.normal(0, sigma, image.shape)
    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image, noise


def deg_from_sim_all(gt_img,
                     sigma_jitter_x_range=[0.5, 2],
                     sigma_jitter_y_range=[0, 0.5],
                     scan_noise_prob=0.2,
                     blur_prob=0.2,
                     pollute_prob=0.5,
                     contrast_prob=0.5,
                     row_line_prob=0.5,
                     blur_kernel_range=[0., 0.],
                     mask_lambda1_range=0.,
                     mask_lambda2_range=0.,
                     contrast_range=[1., 1.],
                     scale_p_range=[1., 1.],
                     sigma_g_range=[1., 1.],
                     row_factor_range=[0.5, 1.5],
                     max_black_pixels=0,
                     max_zinger_pixels=0):
    gt_img = gt_img / 255.0
    out = gt_img.copy()

    # random contrast
    if np.random.uniform() < contrast_prob:
        contrast_factor = np.random.uniform(contrast_range[0], contrast_range[1])
        out = adjust_contrast(out, contrast_factor, maxval=1.)

    # add pollution
    if np.random.uniform() < pollute_prob:
        mask_lambda1 = np.random.uniform(mask_lambda1_range[0], mask_lambda1_range[1])
        mask_lambda2 = np.random.uniform(mask_lambda2_range[0], mask_lambda2_range[1])
        out, _ = add_pollution(out, lamb1=mask_lambda1, lamb2=mask_lambda2)

    # scanning jitter
    if np.random.uniform() < scan_noise_prob:
        sigma_jitter_x = np.random.uniform(sigma_jitter_x_range[0], sigma_jitter_x_range[1])
        sigma_jitter_y = np.random.uniform(sigma_jitter_y_range[0], sigma_jitter_y_range[1])
        jitter_map = generate_jitter_map(out, sigma_jitter_x, sigma_jitter_y, phi_x=0.5, phi_y=0.5)
        out = add_scan_jitter(out, jitter_map[0], jitter_map[1])

    # motion blur
    if np.random.uniform() < blur_prob:
        blur_kernel_size = np.random.randint(blur_kernel_range[0], blur_kernel_range[1])
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        h, w = gt_img.shape
        width = np.random.randint(80, 200)
        row_idx = np.random.randint(0, h - width)
        region = [row_idx, row_idx + width, 0, w]
        out, _ = add_motion_blur(out, kernel_size=blur_kernel_size, region=region)

    # poisson noise
    out, _ = add_poisson_noise(out, lamb_range=scale_p_range)

    # gaussian noise
    out, _ = add_gaussian_noise(out, sigma_range=sigma_g_range)

    # row-line
    if np.random.uniform() < row_line_prob:
        out = add_row_line_noise(out, row_factor_range=row_factor_range)

    # black pixel and zinger pixel
    if np.random.uniform() < 0.5:
        out = add_black_pixel_noise(out, num_points=max_black_pixels)
    if np.random.uniform() < 0.5:
        out = add_zinger_pixel_noise(out, num_points=max_zinger_pixels)

    out = np.clip((out * 255.0).round(), 0, 255)
    gt = np.clip((gt_img * 255.0).round(), 0, 255)
    return out, gt


def mask_augment(mask):
    # flip
    if np.random.rand() < 0.5:
        mask = np.flip(mask, axis=0)
    if np.random.rand() < 0.5:
        mask = np.flip(mask, axis=1)
    if np.random.rand() < 0.5:
        k = np.random.randint(1, 3)
        mask = np.rot90(mask, k)
    return mask


def random_crop(image, shape):
    current_height, current_width = image.shape[:2]
    start_x = np.random.randint(0, current_width - shape[0] + 1)
    start_y = np.random.randint(0, current_height - shape[1] + 1)
    cropped_image = image[start_y:start_y + shape[0], start_x:start_x + shape[1]]
    return cropped_image


def get_mask(mask_folder=r'F:\Datasets\InstructTEMSR\decontamination\mask\local'):
    mask_files = os.listdir(mask_folder)
    mask_file = np.random.choice(mask_files, 1, replace=False)[0]
    mask = cv2.imread(os.path.join(mask_folder, mask_file), 0)
    return mask


def add_pollution(img, lamb1=0.5, lamb2=0.8, mask_pollution=None, mask_background=None):
    if mask_pollution is None:
        mask_pollution = get_mask(r'F:\Datasets\InstructTEMSR\decontamination\mask\local')
        h, w = img.shape
        mask_pollution = random_crop(mask_pollution, (h // 8, w // 8))
        mask_pollution = cv2.resize(mask_pollution, (h, w))
        mask_pollution = mask_pollution / 255.0
    mask_pollution = adjust_contrast(mask_pollution, lamb1, maxval=1.)
    if lamb1 > 0:
        img = np.clip(img * mask_pollution, 0, 1)

    if mask_background is None:
        mask_background = get_mask(r'D:\Datasets\STEMEXP256\bgcnn\pred\purebg')
        mask_background_gradient = get_mask(r'F:\Datasets\InstructTEMSR\decontamination\mask\global\gradient')
        h, w = img.shape
        mask_background = random_crop(mask_background, (h, w))
        mask_background_gradient = np.clip(cv2.resize(mask_background_gradient, (h, w)) / 255.0, 0, 1)
        mask_background = mask_background * mask_background_gradient
        mask_background = mask_background / 255.0
    img = img * lamb2 + mask_background * (1 - lamb2)
    return img, (mask_pollution, mask_background)


def add_motion_blur(image, kernel_size=7, motion_blur_kernel=None, region=None):
    if motion_blur_kernel is None:
        angle = np.random.randint(0, 45)
        M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(kernel_size))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (kernel_size, kernel_size))
        # normalize
        motion_blur_kernel = motion_blur_kernel / np.sum(motion_blur_kernel)
    if region:
        image_region = image[region[0]:region[1], region[2]:region[3]]
        region_size = (region[1] - region[0], region[3] - region[2])
        region_blurred = cv2.filter2D(image_region, -1, motion_blur_kernel)
        image_blurred = image.copy()
        # use gaussian mask to smooth
        mask_g = gaussian_2d(shape=region_size)
        region_orig = image[region[0]:region[1], region[2]:region[3]]
        blurred_content = region_blurred * mask_g + region_orig * (1-mask_g)
        # blurred_content, _ = add_gaussian_noise(blurred_content, sigma_range=[0.02, 0.05])
        image_blurred[region[0]:region[1], region[2]:region[3]] = blurred_content



    else:
        image_blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    image_blurred = np.clip(image_blurred, 0, 1)
    return image_blurred, motion_blur_kernel


def deg_from_sim_denoise(gt_img,
                         sigma_jitter_x_range=[0.5, 1],
                         sigma_jitter_y_range=[0.0, 0.25],
                         scan_noise_prob=0.5,
                         blur_prob=0.5,
                         pollute_prob=0.5,
                         contrast_prob=0.5,
                         row_line_prob=0.5,
                         blur_kernel_range=[7., 21.],
                         mask_lambda1_range=[0.5, 0.9],
                         mask_lambda2_range=[0.7, 1],
                         contrast_range=[0.5, 1.2],
                         scale_p_range=[1., 1.],
                         sigma_g_range=[1., 1.],
                         row_factor_range=[0.5, 1.5],
                         max_black_pixels=0,
                         max_zinger_pixels=0):
    out = gt_img / 255.0
    # random contrast
    if np.random.uniform() < contrast_prob:
        contrast_factor = np.random.uniform(contrast_range[0], contrast_range[1])
        out = adjust_contrast(out, contrast_factor, maxval=1.)

    # motion blur
    if np.random.uniform() < blur_prob:
        blur_kernel_size = np.random.randint(blur_kernel_range[0], blur_kernel_range[1])
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        h, w = gt_img.shape
        width = np.random.randint(80, 200)
        row_idx = np.random.randint(0, h - width)
        region = [row_idx, row_idx + width, 0, w]
        out, _ = add_motion_blur(out, kernel_size=blur_kernel_size, region=region)

    # add pollution
    if np.random.uniform() < pollute_prob:
        mask_lambda1 = np.random.uniform(mask_lambda1_range[0], mask_lambda1_range[1])
        mask_lambda2 = np.random.uniform(mask_lambda2_range[0], mask_lambda2_range[1])
        out, pollute = add_pollution(out, lamb1=mask_lambda1, lamb2=mask_lambda2)

    gt = out.copy()
    # scanning jitter
    if np.random.uniform() < scan_noise_prob:
        sigma_jitter_x = np.random.uniform(sigma_jitter_x_range[0], sigma_jitter_x_range[1])
        sigma_jitter_y = np.random.uniform(sigma_jitter_y_range[0], sigma_jitter_y_range[1])
        jitter_map = generate_jitter_map(out, sigma_jitter_x, sigma_jitter_y, phi_x=0.5, phi_y=0.5)
        out = add_scan_jitter(out, jitter_map[0], jitter_map[1])
        # out, _ = add_scan_noise(out, sigma_jitter, phi=np.pi / 4)

    # poisson noise
    out, _ = add_poisson_noise(out, lamb_range=scale_p_range)
    # gaussian noise
    out, _ = add_gaussian_noise(out, sigma_range=sigma_g_range)

    # row-line
    if np.random.uniform() < row_line_prob:
        out = add_row_line_noise(out, row_factor_range=row_factor_range)
    # black pixel and zinger pixel
    if np.random.uniform() < 0.5:
        out = add_black_pixel_noise(out, num_points=max_black_pixels)
        out = add_zinger_pixel_noise(out, num_points=max_zinger_pixels)

    out = np.clip((out * 255.0).round(), 0, 255)
    gt = np.clip((gt * 255.0).round(), 0, 255)
    return out, gt


def deg_from_sim_ll(gt_img,
                    sigma_jitter_x_range=[0.5, 1],
                    sigma_jitter_y_range=[0.0, 0.25],
                    scan_noise_prob=0.2,
                    blur_prob=0.2,
                    pollute_prob=0.5,
                    contrast_prob=0.5,
                    row_line_prob=0.5,
                    blur_kernel_range=[0., 0.],
                    mask_lambda1_range=[0.5, 0.9],
                    mask_lambda2_range=[0.7, 1],
                    contrast_range=[0.5, 1.2],
                    scale_p_range=[1., 1.],
                    sigma_g_range=[1., 1.],
                    row_factor_range=[0.5, 1.5],
                    max_black_pixels=0,
                    max_zinger_pixels=0):
    out = gt_img / 255.0

    gt = out.copy()
    # # random contrast
    # if np.random.uniform() <= contrast_prob:
    contrast_factor = np.random.uniform(contrast_range[0], contrast_range[1])
    out = adjust_contrast(out, contrast_factor, maxval=1.)
    bright_factor = np.random.uniform(0.3, 0.5)
    out = adjust_brightness(out, bright_factor=bright_factor, maxval=1.)

    # add pollution
    if np.random.uniform() <= pollute_prob:
        mask_lambda1 = np.random.uniform(mask_lambda1_range[0], mask_lambda1_range[1])
        mask_lambda2 = np.random.uniform(mask_lambda2_range[0], mask_lambda2_range[1])
        out, pollute = add_pollution(out, lamb1=mask_lambda1, lamb2=mask_lambda2)
        gt, _ = add_pollution(gt, lamb1=mask_lambda1, lamb2=mask_lambda2, mask_pollution=pollute[0],
                              mask_background=pollute[1])
    # scanning jitter
    if np.random.uniform() <= scan_noise_prob:
        sigma_jitter_x = np.random.uniform(sigma_jitter_x_range[0], sigma_jitter_x_range[1])
        sigma_jitter_y = np.random.uniform(sigma_jitter_y_range[0], sigma_jitter_y_range[1])
        jitter_map = generate_jitter_map(out, sigma_jitter_x, sigma_jitter_y, phi_x=0.5, phi_y=0.5)
        out = add_scan_jitter(out, jitter_map[0], jitter_map[1])
        gt = add_scan_jitter(gt, jitter_map[0], jitter_map[1])
    # motion blur
    if np.random.uniform() <= blur_prob:
        blur_kernel_size = np.random.randint(blur_kernel_range[0], blur_kernel_range[1])
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        out, kernel = add_motion_blur(out, kernel_size=blur_kernel_size)
        gt, _ = add_motion_blur(gt, kernel_size=blur_kernel_size, motion_blur_kernel=kernel)

    # poisson noise
    # out, noise_p = add_poisson_noise(out, lamb_range=scale_p_range)
    # gt, _ = add_poisson_noise(gt, lamb_range=scale_p_range, noise=noise_p * 0.3)

    # gaussian noise
    out, noise_g = add_gaussian_noise(out, sigma_range=sigma_g_range)
    gt, _ = add_gaussian_noise(gt, sigma_range=sigma_g_range, noise=noise_g * 0.2)

    # row-line
    if np.random.uniform() <= row_line_prob:
        out = add_row_line_noise(out, row_factor_range=row_factor_range)
    # black pixel and zinger pixel
    if np.random.uniform() <= 0.5:
        out = add_black_pixel_noise(out, num_points=max_black_pixels)
        out = add_zinger_pixel_noise(out, num_points=max_zinger_pixels)

    out = np.clip((out * 255.0).round(), 0, 255)
    gt = np.clip((gt * 255.0).round(), 0, 255)
    return out, gt


def deg_from_sim_deblur(gt_img,
                        sigma_jitter_x_range=[0.5, 2],
                        sigma_jitter_y_range=[0.25, 0.5],
                        scan_noise_prob=0.2,
                        blur_prob=0.2,
                        pollute_prob=0.5,
                        contrast_prob=0.5,
                        row_line_prob=0.5,
                        blur_kernel_range=[0., 0.],
                        mask_lambda1_range=[0.5, 0.9],
                        mask_lambda2_range=[0.7, 1],
                        contrast_range=[0.5, 1.2],
                        scale_p_range=[1., 1.],
                        sigma_g_range=[1., 1.],
                        row_factor_range=[0.5, 1.5],
                        max_black_pixels=0,
                        max_zinger_pixels=0):
    out = gt_img / 255.0
    # # random contrast
    # if np.random.uniform() < contrast_prob:
    #     contrast_factor = np.random.uniform(contrast_range[0], contrast_range[1])
    #     out = adjust_contrast(out, contrast_factor, maxval=1.)
    #
    # add pollution
    if np.random.uniform() < pollute_prob:
        mask_lambda1 = np.random.uniform(mask_lambda1_range[0], mask_lambda1_range[1])
        mask_lambda2 = np.random.uniform(mask_lambda2_range[0], mask_lambda2_range[1])
        out, _ = add_pollution(out, lamb1=mask_lambda1, lamb2=mask_lambda2)

    gt = out.copy()



    # motion blur
    if np.random.uniform() < blur_prob:
        blur_kernel_size = np.random.randint(blur_kernel_range[0], blur_kernel_range[1])
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        h, w = gt_img.shape
        width = np.random.randint(80, 250)
        row_idx = np.random.randint(0, h - width)
        region = [row_idx, row_idx + width, 0, w]
        out, _ = add_motion_blur(out, kernel_size=blur_kernel_size, region=region)

    # scanning jitter
    if np.random.uniform() < scan_noise_prob:
        sigma_jitter_x = np.random.uniform(sigma_jitter_x_range[0], sigma_jitter_x_range[1])
        sigma_jitter_y = np.random.uniform(sigma_jitter_y_range[0], sigma_jitter_y_range[1])
        jitter_map = generate_jitter_map(out, sigma_jitter_x, sigma_jitter_y, phi_x=0.5, phi_y=0.5)
        out = add_scan_jitter(out, jitter_map[0], jitter_map[1])

    # poisson noise
    if scale_p_range[1] != 0:
        out, p_noise = add_poisson_noise(out, lamb_range=scale_p_range)
        gt, _ = add_poisson_noise(gt, lamb_range=scale_p_range, noise=p_noise)

    # gaussian noise
    out, g_noise = add_gaussian_noise(out, sigma_range=sigma_g_range)

    # # row-line
    # if np.random.uniform() < row_line_prob:
    #     out = add_row_line_noise(out, row_factor_range=row_factor_range)
    # # black pixel and zinger pixel
    # if np.random.uniform() < 0.5:
    #     out = add_black_pixel_noise(out, num_points=max_black_pixels)
    #     out = add_zinger_pixel_noise(out, num_points=max_zinger_pixels)

    out = np.clip((out * 255.0).round(), 0, 255)
    gt = np.clip((gt * 255.0).round(), 0, 255)
    return out, gt


def deg_from_sim_decontamination(gt_img,
                                 sigma_jitter_range=[2, 2],
                                 scan_noise_prob=0.2,
                                 blur_prob=0.2,
                                 pollute_prob=0.5,
                                 contrast_prob=0.5,
                                 row_line_prob=0.5,
                                 blur_kernel_range=[0., 0.],
                                 mask_lambda1_range=[0.5, 0.9],
                                 mask_lambda2_range=[0.7, 1],
                                 contrast_range=[0.5, 1.2],
                                 scale_p_range=[1., 1.],
                                 sigma_g_range=[1., 1.],
                                 row_factor_range=[0.5, 1.5],
                                 max_black_pixels=0,
                                 max_zinger_pixels=0):
    out = gt_img / 255.0
    # random contrast
    if np.random.uniform() < contrast_prob:
        contrast_factor = np.random.uniform(contrast_range[0], contrast_range[1])
        out = adjust_contrast(out, contrast_factor, maxval=1.)
    gt = out.copy()
    # add pollution
    if np.random.uniform() < pollute_prob:
        mask_lambda1 = np.random.uniform(mask_lambda1_range[0], mask_lambda1_range[1])
        mask_lambda2 = np.random.uniform(mask_lambda2_range[0], mask_lambda2_range[1])
        out, _ = add_pollution(out, lamb1=mask_lambda1, lamb2=mask_lambda2)

    # scanning jitter
    if np.random.uniform() < scan_noise_prob:
        sigma_jitter = np.random.uniform(sigma_jitter_range[0], sigma_jitter_range[1])
        out, delta_map = add_scan_noise(out, sigma_jitter, phi=np.pi / 4)
        gt, _ = add_scan_noise(gt, sigma_jitter, phi=np.pi / 4, delta_map_x=delta_map[0], delta_map_y=delta_map[1])

    # motion blur
    if np.random.uniform() < blur_prob:
        blur_kernel_size = np.random.randint(blur_kernel_range[0], blur_kernel_range[1])
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        out, kernel = add_motion_blur(out, kernel_size=blur_kernel_size)
        gt, _ = add_motion_blur(gt, kernel_size=blur_kernel_size, motion_blur_kernel=kernel)

    # poisson noise
    out, p_noise = add_poisson_noise(out, lamb_range=scale_p_range)
    gt, _ = add_poisson_noise(gt, lamb_range=scale_p_range, noise=p_noise * 0.3)
    # gaussian noise
    out, g_noise = add_gaussian_noise(out, sigma_range=sigma_g_range)
    gt, _ = add_gaussian_noise(gt, sigma_range=sigma_g_range, noise=g_noise * 0.2)

    # row-line
    if np.random.uniform() < row_line_prob:
        out = add_row_line_noise(out, row_factor_range=row_factor_range)
    # black pixel and zinger pixel
    if np.random.uniform() < 0.5:
        out = add_black_pixel_noise(out, num_points=max_black_pixels)
        out = add_zinger_pixel_noise(out, num_points=max_zinger_pixels)

    out = np.clip((out * 255.0).round(), 0, 255)
    gt = np.clip((gt * 255.0).round(), 0, 255)
    return out, gt


def deg_from_exp_sr(gt_img,
                    resize_range=[0.25, 3.0],
                    sigma_g_range=[1., 1.],
                    resize_prob=[0.5, 0.5, 0.0]):
    out = gt_img / 255.0

    out = torch.from_numpy(out).unsqueeze(0).unsqueeze(0)

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
    out = F.interpolate(out, size=(gt_img.shape[0], gt_img.shape[1]), mode=mode)
    out = out.detach().cpu().numpy().squeeze()
    # gaussian noise
    out, g_noise = add_gaussian_noise(out, sigma_range=sigma_g_range)

    out = np.clip((out * 255.0).round(), 0, 255)
    gt = gt_img
    return out, gt


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


def generate_jitter_map(img, sigma_x, sigma_y, phi_x, phi_y, local=False):
    h, w = img.shape
    map_size = (h, w)
    delta_map_x = np.zeros(shape=map_size[1], dtype=float)
    delta_map_y = np.zeros(shape=map_size[0], dtype=float)
    if local:
        length = np.random.randint(low=w // 4, high=w // 2)
        start_x = np.random.randint(low=0, high=w - w // 4)
        local_range_x = [x for x in range(start_x, start_x + length)]
        print(local_range_x)
        local_range_y = [y for y in range(h)]
    else:
        local_range_x = [x for x in range(w)]
        local_range_y = [y for y in range(h)]

    for i in range(map_size[1]):
        if i in local_range_x:
            if i == local_range_x[0]:
                alpha_t = np.random.normal(0, sigma_x)
                delta_map_x[i] = alpha_t / np.sqrt(1 - phi_x ** 2)
            else:
                alpha_t = np.random.normal(0, sigma_x)
                delta_map_x[i] = phi_x * delta_map_x[i - 1] + alpha_t

    for j in range(map_size[0]):
        if j in local_range_y:
            if j == local_range_y[0]:
                alpha_t = np.random.normal(0, sigma_y)
                delta_map_y[j] = alpha_t / np.sqrt(1 - phi_y ** 2)
            else:
                alpha_t = np.random.normal(0, sigma_y)
                delta_map_y[j] = phi_y * delta_map_y[j - 1] + alpha_t
    delta_map_x = delta_map_x.round().astype(int)
    delta_map_y = delta_map_y.round().astype(int)
    return delta_map_x, delta_map_y


def add_scan_jitter(img, delta_map_x, delta_map_y):
    h, w = img.shape
    img_pix = img.squeeze()
    img_new = img.copy()
    for i in range(h):
        for j in range(w):
            try:
                img_new[i][j] = img_pix[i - delta_map_x[i]][j - delta_map_y[j]]
            except:
                img_new[i][j] = img_pix[i][j]

    return img_new


def add_scan_noise(img, sigma_jitter=0.2, phi=np.pi / 4, f=1 / 20, delta_map_x=None, delta_map_y=None):
    h, w = img.shape
    img_pix = img.squeeze()
    img_new = img.copy()
    if (delta_map_x is None) or (delta_map_y is None):
        delta_map_x = np.zeros(shape=h, dtype=int)
        delta_map_y = np.zeros(shape=(h, w), dtype=int)
        for i in range(h):
            delta_x = int((np.random.normal() * sigma_jitter * np.sin(2 * np.pi * f * i)).round())
            delta_map_x[i] = delta_x
            for j in range(w):
                delta_y = int((np.random.normal() * sigma_jitter * np.sin(2 * np.pi * f * j + phi)).round())
                delta_map_y[i][j] = delta_y
                try:
                    img_new[i][j] = img_pix[i - delta_x][j - delta_y]
                except:
                    img_new[i][j] = img_pix[i][j]
    else:
        for i in range(h):
            delta_x = delta_map_x[i]
            for j in range(w):
                delta_y = delta_map_y[i][j]
                try:
                    img_new[i][j] = img_pix[i - delta_x][j - delta_y]
                except:
                    img_new[i][j] = img_pix[i][j]
    return img_new, (delta_map_x, delta_map_y)


def make_deg_folder(n_thread, orig_folder,
                    save_gt_folder,
                    save_lq_folder,
                    mode,
                    params,
                    repeats):
    from multiprocessing import Pool
    from tqdm import tqdm

    img_list = os.listdir(orig_folder)
    if n_thread != 1:
        pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
        pool = Pool(n_thread)
    for idx, filename in enumerate(img_list):
        orig_path = os.path.join(orig_folder, filename)
        if save_gt_folder:
            save_gt_path = os.path.join(save_gt_folder, filename)
        else:
            save_gt_path = None
        save_lq_path = os.path.join(save_lq_folder, filename)
        if n_thread == 1:
            worker(idx, orig_path, save_gt_path, save_lq_path, repeats, mode, params)  # single thread
        else:
            pool.apply_async(worker, args=(idx, orig_path, save_gt_path, save_lq_path, repeats, mode, params),
                             callback=lambda arg: pbar.update(1))
    if n_thread != 1:
        pool.close()
        pool.join()
        pbar.close()
    print('All processes done.')


def worker(idx, orig_path, save_gt_path, save_lq_path, repeats, mode, params):
    setup_seed(idx + seed)
    img_sim = cv2.imread(orig_path, 0)
    # img_sim_pt = grayimg2tensor(img_sim)
    if mode == 'denoise':
        func = deg_from_sim_denoise
    elif mode == 'deblur':
        func = deg_from_sim_deblur
    elif mode == 'decontamination':
        func = deg_from_sim_decontamination
    elif mode == 'll':
        func = deg_from_sim_ll
    elif mode == 'seg':
        func = deg_from_sim_all
    elif mode == 'sr':
        func = deg_from_exp_sr
    else:
        raise NotImplementedError

    for it in range(repeats):
        if func != deg_from_exp_sr:
            img_deg, img_gt = func(img_sim,
                                   sigma_jitter_x_range=params['sigma_jitter_x_range'],
                                   sigma_jitter_y_range=params['sigma_jitter_y_range'],
                                   scan_noise_prob=params['scan_noise_prob'],
                                   blur_prob=params['blur_prob'],
                                   pollute_prob=params['pollute_prob'],
                                   row_line_prob=params['row_line_prob'],
                                   blur_kernel_range=params['blur_kernel_range'],
                                   mask_lambda1_range=params['mask_lambda1_range'],
                                   mask_lambda2_range=params['mask_lambda2_range'],
                                   contrast_range=params['contrast_range'],
                                   scale_p_range=params['scale_p_range'],
                                   sigma_g_range=params['sigma_g_range'],
                                   row_factor_range=params['row_factor_range'],
                                   max_black_pixels=params['max_black_pixels'],
                                   max_zinger_pixels=params['max_zinger_pixels'])
        else:
            img_deg, img_gt = func(img_sim,
                                   resize_range=[0.25, 4.0],
                                   sigma_g_range=[0.01, 0.05],
                                   resize_prob=[0.5, 0.5, 0.0]
                                   )
        if repeats != 1:
            cv2.imwrite(save_lq_path.replace('.png', '_{}.png'.format(it + 1)), img_deg, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            if save_gt_path:
                if mode == 'seg':
                    filename = os.path.split(orig_path)[1]
                    shutil.copy(os.path.join(r'F:\Datasets\TEM-ImageNet-v1.3-master\noNoiseNoBackgroundSuperresolution',
                                             filename),
                                os.path.join(save_gt_path, filename))
                cv2.imwrite(save_gt_path.replace('.png', '_{}.png'.format(it + 1)), img_gt, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            cv2.imwrite(save_lq_path, img_deg, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            if save_gt_path:
                if mode == 'seg':
                    filename = os.path.split(orig_path)[1]
                    shutil.copy(os.path.join(r'F:\Datasets\TEM-ImageNet-v1.3-master\noNoiseNoBackgroundSuperresolution',
                                             filename),
                                save_gt_path)
                else:
                    cv2.imwrite(save_gt_path, img_gt, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == '__main__':
    import json

    # setup_seed(12345)
    # with open('./deg_params/params_all.json', 'r') as f:
    #     params = json.load(f)
    # name = 'TEMImageNet1000'
    # task = 'Segmentation'
    # mode = 'seg'
    # make_deg_folder(n_thread=8, orig_folder=r'F:\Datasets\InstructSTEMIR\Original\TEMImageNet1000',
    #                 save_gt_folder=r'F:\Datasets\InstructSTEMIR\{}\GT\{}'.format(task, name),
    #                 save_lq_folder=r'F:\Datasets\InstructSTEMIR\{}\LQ\{}'.format(task, name),
    #                 repeats=1,
    #                 mode=mode,
    #                 params=params)


    with open(r'./deg_params/params_denoise_exp.json', 'r') as f:
        params = json.load(f)

    '''
    make_deg_folder(n_thread=2, orig_folder=r'F:\Datasets\InstructTEMSR\All_GT\TEMImageNet_sub500',
                    save_gt_folder=None,
                    save_lq_folder=r'F:\Datasets\InstructSTEMIR\test',
                    mode='denoise',
                    repeats=1,
                    params=params)
    '''

    make_deg_folder(n_thread=6, orig_folder=r'F:\Datasets\InstSIP_test\Denoise\GT',
                    save_gt_folder=None,
                    save_lq_folder=r'F:\Datasets\InstSIP_test\Denoise\LQ',
                    repeats=1,
                    mode='denoise',
                    params=params)


