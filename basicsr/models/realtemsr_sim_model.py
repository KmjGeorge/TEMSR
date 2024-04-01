import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch.nn import functional as F

from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from scripts.data_preparation.STEMdeg import adjust_contrast


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


def add_heteroscedastic_gnoise(image, device, sigma_1_range=(0.05, 0.5), sigma_2_range=(0.01, 0.1)):
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


def add_scan_noise(img, sigma_jitter=0.2, phi=np.pi / 4, f=1 / 200):
    _, _, h, w = img.shape
    img_new = torch.zeros_like(img)
    for i in range(h):
        delta_x = int((np.random.normal() * sigma_jitter * np.sin(2 * np.pi * f * i)).round())
        for j in range(w):
            delta_y = int((np.random.normal() * sigma_jitter * np.sin(2 * np.pi * f * j + phi)).round())
            try:
                img_new[:, :, i, j] = img[:, :, i - delta_x, j - delta_y]
            except:
                img_new[:, :, i, j] = img[:, :, i, j]
    return img_new


@MODEL_REGISTRY.register()
class RealTEMSRSimModel(SRModel):
    """RealESRNet Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(RealTEMSRSimModel, self).__init__(opt)
        self.queue_size = opt.get('queue_size', 180)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        if self.is_train and self.opt.get('degradation', True):
            # training data synthesis

            self.gt = data['gt'].to(self.device)
            self.kernel = data['kernel'].to(self.device)
            ori_h, ori_w = self.gt.size()[2:4]

            out = self.gt
            if self.opt['randomshuffle']:
                seq = np.random.choice([1, 2, 3, 4, 5, 6], size=6, replace=False)
            else:
                seq = [1, 2, 3, 4, 5, 6]

            for step in seq:
                if step == 1:
                    if np.random.uniform() < self.opt['blur_prob']:
                        out = filter2D(self.gt, self.kernel)
                elif step == 2:
                    # adaptive luminance adjustment
                    # thresh_factor = np.random.uniform(self.opt['threshold_factor_range'][0],
                    #                                   self.opt['threshold_factor_range'][1])
                    # out = adaptive_luminance_adjust(out, self.opt['neighborhood_size'], thresh_factor)
                    # global contrast adjustment
                    contrast_factor = np.random.uniform(self.opt['contrast_range'][0], self.opt['contrast_range'][1])
                    out = adjust_contrast(out, contrast_factor=contrast_factor)
                elif step == 3:
                    # exposure compensation
                    compensation_value = np.random.uniform(self.opt['compensation_range'][0],
                                                           self.opt['compensation_range'][1])
                    out = linear_exposure_compensation(out, maxval=1., compensation_value=compensation_value)
                elif step == 4:
                    # random resize
                    updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
                    if updown_type == 'up':
                        scale = np.random.uniform(1, self.opt['resize_range'][1])
                    elif updown_type == 'down':
                        scale = np.random.uniform(self.opt['resize_range'][0], 1)
                    else:
                        scale = 1
                    mode = random.choice(['area', 'bilinear', 'bicubic'])
                    out = F.interpolate(out, scale_factor=scale, mode=mode)
                    # resize back
                    mode = random.choice(['area', 'bilinear', 'bicubic'])
                    out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                    # out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
                elif step == 5:
                    # add noise
                    # 1. poisson noise
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt['poisson_scale_range'],
                        gray_prob=1,
                        clip=True,
                        rounds=False)
                    # 2. gaussian noise
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=self.opt['gaussian_sigma_range'], clip=True, rounds=False, gray_prob=1)
                elif step == 6:
                    out = add_scan_noise(out, self.opt['sigma_jitter'], phi=np.pi / 2)

            self.lq = torch.clamp((out[:, 0, :, :].unsqueeze(1) * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['gt_size']
            self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size, self.opt['scale'])

            '''
            for i in range(self.gt.shape[0]):
                gt_img = (self.gt.detach().cpu().numpy()[i].squeeze() * 255).astype(np.uint8)
                lq_img = (self.lq.detach().cpu().numpy()[i].squeeze() * 255).astype(np.uint8)

                cv2.imwrite('D:\github\TEMSR\experiments\\gt{}.png'.format(i), gt_img)
                cv2.imwrite('D:\github\TEMSR\experiments\\lq{}.png'.format(i), lq_img)
                # plt.subplot(121)
                # plt.title('gt')
                # plt.imshow(gt_img, 'gray')
                # plt.subplot(122)
                # plt.title('lq')
                # plt.imshow(lq_img, 'gray')
                # plt.show()
            assert False
            '''

            # training pair pool
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

        else:
            # for paired training or validation
            # to single channel
            self.lq = data['lq'][:, 0, :, :].unsqueeze(1).to(self.device)
            if 'gt' in data:
                self.gt = data['gt'][:, 0, :, :].unsqueeze(1).to(self.device)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(RealTEMSRSimModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True
