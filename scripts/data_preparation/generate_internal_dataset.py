import os.path

import PIL
import numpy as np
import sys
import random
import torch
from basicsr import set_random_seed
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import cv2
import numpy as np
import PIL
import random
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers


class RandomRotationFromSequence(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = int(np.random.choice(degrees))
        return angle

    def __call__(self, data):
        """
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """
        hr, lr = data
        angle = self.get_params(self.degrees)
        return F.rotate(hr, angle, self.resample, self.expand, self.center), \
               F.rotate(lr, angle, self.resample, self.expand, self.center)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a probability of 0.5."""

    def __call__(self, data):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        hr, lr = data
        if random.random() < 0.5:
            return F.hflip(hr), F.hflip(lr)
        return hr, lr


class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a probability of 0.5."""

    def __call__(self, data):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        hr, lr = data
        if random.random() < 0.5:
            return F.vflip(hr), F.vflip(lr)
        return hr, lr


class RandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(data, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        hr, lr = data
        w, h = hr.size
        th, tw = output_size
        if w == tw or h == th:
            return 0, 0, h, w

        if w < tw or h < th:
            th, tw = h // 2, w // 2

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, data):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        hr, lr = data
        if self.padding > 0:
            hr = F.pad(hr, self.padding)
            lr = F.pad(lr, self.padding)

        i, j, h, w = self.get_params(data, self.size)
        return F.crop(hr, i, j, h, w), F.crop(lr, i, j, h, w)


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, data):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        hr, lr = data
        return F.to_tensor(hr), F.to_tensor(lr)


class DataSampler:
    def __init__(self, img, sr_factor, crop_size):
        self.img = img
        self.sr_factor = sr_factor
        self.pairs = self.create_hr_lr_pairs()
        sizes = np.float32([x[0].size[0] * x[0].size[1] / float(img.size[0] * img.size[1])
                            for x in self.pairs])
        self.pair_probabilities = sizes / np.sum(sizes)

        self.transform = transforms.Compose([
            RandomRotationFromSequence([0, 90, 180, 270]),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomCrop(crop_size),
            ToTensor()])

    def create_hr_lr_pairs(self):
        smaller_side = min(self.img.size[0: 2])
        larger_side = max(self.img.size[0: 2])
        factors = []
        for i in range(smaller_side // 5, smaller_side + 1):
            downsampled_smaller_side = i
            zoom = float(downsampled_smaller_side) / smaller_side
            downsampled_larger_side = round(larger_side * zoom)
            if downsampled_smaller_side % self.sr_factor == 0 and downsampled_larger_side % self.sr_factor == 0:
                factors.append(zoom)
        pairs = []
        for zoom in factors:
            hr = self.img.resize((int(self.img.size[0] * zoom),
                                  int(self.img.size[1] * zoom)),
                                 resample=PIL.Image.BICUBIC)

            lr = hr.resize((int(hr.size[0] / self.sr_factor),
                            int(hr.size[1] / self.sr_factor)),
                           resample=PIL.Image.BICUBIC)

            # lr = lr.resize(hr.size, resample=PIL.Image.BICUBIC)
            print(lr.size, hr.size)
            pairs.append((hr, lr))

        return pairs

    def generate_data(self):
        while True:
            hr, lr = random.choices(self.pairs, weights=self.pair_probabilities, k=1)[0]
            hr_tensor, lr_tensor = self.transform((hr, lr))
            hr_tensor = torch.unsqueeze(hr_tensor, 0)
            lr_tensor = torch.unsqueeze(lr_tensor, 0)
            yield hr_tensor, lr_tensor


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import yaml
    from PIL import Image
    with open('internal_dataset.yml', 'r', encoding='utf-8') as f:
        opt = yaml.load(f.read(), Loader=yaml.FullLoader)

    set_random_seed(opt['manual_seed'])
    img = PIL.Image.open(opt['img_path'])
    scale = opt['scale']
    crop = opt['crop_size']
    iter = opt['iter']
    filename = os.path.split(opt['img_path'])[1]
    save_path = opt['save_path']
    sampler = DataSampler(img, scale, crop)

    i = 0
    for x in sampler.generate_data():
        if i == iter:
            break
        hr, lr = x
        hr = hr.numpy()[0][0] * 255
        lr = lr.numpy()[0][0] * 255
        hr_img = Image.fromarray(hr).convert('L')
        lr_img = Image.fromarray(lr).convert('L')
        hr_img.save(os.path.join(save_path, 'GT', filename.replace('.png', '_{}.png').format(i+1)))
        lr_img.save(os.path.join(save_path, 'LQ', filename.replace('.png', '_{}.png').format(i+1)))
        # plt.subplot(121)
        # plt.title('HR')
        # plt.imshow(hr, 'gray')
        # plt.subplot(122)
        # plt.title('LR')
        # plt.imshow(lr, 'gray')
        # plt.show()
        i += 1


