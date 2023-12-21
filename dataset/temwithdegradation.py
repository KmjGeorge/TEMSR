import cv2
import math

import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image


@DATASET_REGISTRY.register()
class RealESRGANDataset(data.Dataset):
    """Dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(RealESRGANDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']
        self.paths = [os.path.join(self.gt_folder, v) for v in os.listdir(opt['dataroot_gt'])]

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.crop_pad_size = opt['crop_pad_size']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes, float32=True)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        # crop or pad to 400
        # TODO: 400 is hard-coded. You may change it accordingly
        h, w = img_gt.shape[0:2]
        crop_pad_size = self.crop_pad_size
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'gt_path': gt_path}
        return return_d

    def __len__(self):
        return len(self.paths)


def pictoshow(pic):
    if len(pic.shape) == 3:
        pic = np.swapaxes(pic, 0, 1)
        pic = np.swapaxes(pic, 1, 2)
    pic = np.uint8(pic * 255)
    return pic


def high_order_degradation(data, opt):
    jpeger = DiffJPEG(differentiable=False)  # simulate JPEG compression artifacts
    usm_sharpener = USMSharp()  # do usm sharpening

    gt = data['gt']
    gt_usm = usm_sharpener(gt)

    kernel1 = data['kernel1']
    kernel2 = data['kernel2']
    sinc_kernel = data['sinc_kernel']
    ori_h, ori_w = gt.size()[2:4]

    # ----------------------- The first degradation process ----------------------- #
    # blur
    out = filter2D(gt_usm, kernel1)
    # random resize
    updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, opt['resize_range'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(opt['resize_range'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scale, mode=mode)
    # add noise
    gray_noise_prob = opt['gray_noise_prob']
    if np.random.uniform() < opt['gaussian_noise_prob']:
        out = random_add_gaussian_noise_pt(
            out, sigma_range=opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=opt['poisson_scale_range'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)
    # JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range'])
    out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
    out = jpeger(out, quality=jpeg_p)

    # ----------------------- The second degradation process ----------------------- #
    # blur
    if np.random.uniform() < opt['second_blur_prob']:
        out = filter2D(out, kernel2)
    # random resize
    updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob2'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, opt['resize_range2'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(opt['resize_range2'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(
        out, size=(int(ori_h / opt['scale'] * scale), int(ori_w / opt['scale'] * scale)), mode=mode)
    # add noise
    gray_noise_prob = opt['gray_noise_prob2']
    if np.random.uniform() < opt['gaussian_noise_prob2']:
        out = random_add_gaussian_noise_pt(
            out, sigma_range=opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=opt['poisson_scale_range2'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)

    # JPEG compression + the final sinc filter
    # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
    # as one operation.
    # We consider two orders:
    #   1. [resize back + sinc filter] + JPEG compression
    #   2. JPEG compression + [resize back + sinc filter]
    # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
    if np.random.uniform() < 0.5:
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode=mode)
        out = filter2D(out, sinc_kernel)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
    else:
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode=mode)
        out = filter2D(out, sinc_kernel)

    # clamp and round
    lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

    # random crop
    gt_size = opt['gt_size']
    (gt, gt_usm), lq = paired_random_crop([gt, gt_usm], lq, gt_size, opt['scale'])

    return gt.detach().numpy(), gt_usm.detach().numpy(), lq.detach().numpy()



def bicubic_lq():
    path = 'G:/datasets/TEMPatch for SR/LQ/Val/'
    target_path = 'G:/datasets/TEMPatch for SR/LQ_Bic/Val/'
    for filename in os.listdir(path):
        img = Image.open(os.path.join(path, filename))
        h, w = img.size
        img = img.resize((h*2, w*2), Image.BICUBIC)
        img.save(os.path.join(target_path, filename))

if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    import yaml
    from PIL import Image
    from tqdm import tqdm
    batch_size = 2
    matplotlib.use('TkAgg')
    with open('../configs/train_realesrgan_x2plus.yml', 'r', encoding='utf-8') as f:
        opt = yaml.load(f.read(), Loader=yaml.FullLoader)
    dataset = RealESRGANDataset(opt['datasets']['train'])
    datalodaer = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    i = 1
    max_iter = 10
    for iter in range(max_iter):
        for x in tqdm(datalodaer):
            gt, gt_usm, lq = high_order_degradation(x, opt)
            for j in range(batch_size):
                # gt_orig = x['gt'][i]  # (3, 512, 512)
                # kernel1 = x['kernel1'][i]
                # kernel2 = x['kernel2'][i]
                # sinc_kernel = x['sinc_kernel'][i]
                gt_img = pictoshow(gt[j])
                gt_img = Image.fromarray(gt_img)
                gt_img.save('G:/datasets/TEMPatch for SR/GT/{}_gt.png'.format(i))

                lq_img = pictoshow(lq[j])
                lq_img = Image.fromarray(lq_img)
                lq_img.save('G:/datasets/TEMPatch for SR/LQ/{}_lq.png'.format(i))
                i += 1
                # # plt.imshow(pictoshow(gt))
                # plt.subplot(241)
                # plt.title('kernel1')
                # plt.imshow(pictoshow(kernel1))
                # plt.subplot(242)
                # plt.title('kernel2')
                # plt.imshow(pictoshow(kernel2))
                # plt.subplot(243)
                # plt.title('sinc_kernel')
                # plt.imshow(pictoshow(sinc_kernel))
                # plt.subplot(245)
                # plt.title('gt')
                # plt.imshow(pictoshow(gt))
                #
                # gt, gt_usm, lq = high_order_degradation(x, opt)
                # plt.subplot(246)
                # plt.title('gt_new')
                # print(gt.shape, lq.shape)
                # plt.imshow(pictoshow(gt[0]))
                # plt.subplot(247)
                # plt.title('gt_new_usm')
                # plt.imshow(pictoshow(gt_usm[0]))
                # plt.subplot(248)
                # plt.title('lq')
                # plt.imshow(pictoshow(lq[0]))
                #
                # plt.show()

