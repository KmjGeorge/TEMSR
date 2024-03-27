import numpy as np
import random
import torch

from basicsr.models.sr_model import SRModel
from torch.nn import functional as F
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.source_target_transforms import RandomRotationFromSequence, RandomHorizontalFlip, RandomVerticalFlip, \
    RandomCrop
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

@MODEL_REGISTRY.register()
class InternalModel(SRModel):
    def __init__(self, opt):
        super(InternalModel, self).__init__(opt)
        if opt['datasets'].get('train') is not None:
            self.use_deg = opt['datasets']['train'].get('use_degradation', False)
        else:
            self.use_deg = False
        if self.use_deg:
            self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
            self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.crop_size = opt.get('crop_size')
        self.sr_factor = opt.get('scale')
        self.transform = transforms.Compose([
            RandomRotationFromSequence([0, 90, 180, 270]),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomCrop(self.crop_size, self.sr_factor)])
        self.pairs_register = None
        self.pair_probabilities_register = None

    def _create_hr_lr_pairs(self, data):
        self.img = data['lq']
        self.h, self.w = self.img.size()[2:4]
        smaller_side = min(self.h, self.w)
        larger_side = max(self.h, self.w)
        factors = []
        for i in range(smaller_side // 5, smaller_side + 1):
            downsampled_smaller_side = i
            zoom = float(downsampled_smaller_side) / smaller_side
            downsampled_larger_side = round(larger_side * zoom)
            if downsampled_smaller_side % self.sr_factor == 0 and downsampled_larger_side % self.sr_factor == 0:
                factors.append(zoom)

        pairs = []

        for zoom in factors:
            hr = F.interpolate(self.img, size=(int(self.h * zoom), int(self.w * zoom)), mode='bicubic')
            lr = F.interpolate(hr, size=(int(hr.shape[2] / self.sr_factor), int(hr.shape[3] / self.sr_factor)), mode='bicubic')
            pairs.append((hr, lr))

        return pairs

    @torch.no_grad()
    def feed_data(self, data):
        if self.is_train:
            if self.pairs_register is None:
                self.pairs = self._create_hr_lr_pairs(data)
                self.pairs_register = self.pairs.copy()
                sizes = np.float32([x[0].size()[2] * x[0].size()[3] / float(self.h * self.w) for x in self.pairs])
                self.pair_probabilities = sizes / np.sum(sizes)
                self.pair_probabilities_register = self.pair_probabilities.copy()
            else:
                self.pairs = self.pairs_register
                self.pair_probabilities = self.pair_probabilities_register

            hr, lr = random.choices(self.pairs, weights=self.pair_probabilities, k=1)[0]
            hr_augmented, lr_augmented = self.transform((hr, lr))

            # avoid little crops
            lr_augmented = F.interpolate(lr_augmented, size=(self.crop_size // self.sr_factor, self.crop_size // self.sr_factor), mode='bicubic')
            hr_augmented = F.interpolate(hr_augmented, size=(self.crop_size, self.crop_size), mode='bicubic')

            self.gt = hr_augmented.to(self.device)
            if not self.use_deg:
                self.lq = lr_augmented.to(self.device)
            else:
                self.kernel1 = data['kernel1'].to(self.device)
                self.kernel2 = data['kernel2'].to(self.device)
                self.sinc_kernel = data['sinc_kernel'].to(self.device)

                # ----------------------- The first degradation process ----------------------- #
                # blur
                out = filter2D(lr_augmented, self.kernel1)

                # add noise
                gray_noise_prob = self.opt['gray_noise_prob']
                if np.random.uniform() < self.opt['gaussian_noise_prob']:
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt['poisson_scale_range'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
                out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
                out = self.jpeger(out, quality=jpeg_p)

                # ----------------------- The second degradation process ----------------------- #
                # blur
                if np.random.uniform() < self.opt['second_blur_prob']:
                    out = filter2D(out, self.kernel2)

                # add noise
                gray_noise_prob = self.opt['gray_noise_prob2']
                if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt['poisson_scale_range2'],
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
                    # the final sinc filter
                    out = filter2D(out, self.sinc_kernel)
                    # JPEG compression
                    jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                    out = torch.clamp(out, 0, 1)
                    out = self.jpeger(out, quality=jpeg_p)
                else:
                    # JPEG compression
                    jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                    out = torch.clamp(out, 0, 1)
                    out = self.jpeger(out, quality=jpeg_p)
                    # the final sinc filter
                    out = filter2D(out, self.sinc_kernel)

                # clamp and round
                self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
                self.lq = self.lq.contiguous().to(self.device)  # for the warning: grad and param do not obey the gradient layout contract
        else:
           self.lq = data['lq'].to(self.device)
           if 'gt' in data:
               self.gt = data['gt'].to(self.device)

        # print(self.use_deg, self.gt.shape, self.lq.shape)
        # hr_show = self.gt[0].permute(1, 2, 0).cpu().numpy()
        # lr_show = self.lq[0].permute(1, 2, 0).cpu().numpy()
        # plt.subplot(121)
        # plt.title('hr')
        # plt.imshow(hr_show, 'gray')
        # plt.subplot(122)
        # plt.title('lr')
        # plt.imshow(lr_show, 'gray')
        # plt.show()
        # assert False
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(InternalModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

