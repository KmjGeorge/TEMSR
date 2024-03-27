from os import path as osp
import torch
from torch.utils import data as data
import numpy as np
import random
from torchvision.transforms.functional import normalize
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.data_util import paths_from_lmdb
from basicsr.utils import FileClient, imfrombytes, img2tensor, rgb2ycbcr, scandir
from basicsr.utils.registry import DATASET_REGISTRY
import math


@DATASET_REGISTRY.register()
class InternalDataset(data.Dataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(InternalDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.lq_folder = opt['dataroot_lq']
        self.use_deg = opt['use_degradation']
        if self.use_deg:
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

            self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21

            self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
            self.pulse_tensor[10, 10] = 1

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder]
            self.io_backend_opt['client_keys'] = ['lq']
            self.paths = paths_from_lmdb(self.lq_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.lq_folder, line.rstrip().split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.lq_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load lq image
        lq_path = self.paths[index]
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)

        if self.use_deg:
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
            kernel = torch.FloatTensor(kernel)
            kernel2 = torch.FloatTensor(kernel2)
            return {'lq': img_lq, 'lq_path': lq_path, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel}
        else:
            return {'lq': img_lq, 'lq_path': lq_path}

    def __len__(self):
        return len(self.paths)
