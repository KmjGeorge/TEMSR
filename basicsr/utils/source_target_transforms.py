import matplotlib.pyplot as plt
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

    def __init__(self, size, sr_factor=2, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.sr_factor = sr_factor

    def get_params(self, data, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        hr, lr = data
        w, h = hr.size()[2:4]

        th, tw = output_size
        if w == tw or h == th:
            return 0, 0, h, w

        if w < tw or h < th:
            th, tw = h // 2, w // 2

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        if i % self.sr_factor != 0:
            i -= i % self.sr_factor
            if i <= 0:
                i = 0
        if j % self.sr_factor != 0:
            j -= j % self.sr_factor
            if j <= 0:
                j = 0
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
        # print(i, j, h, w)
        if h % self.sr_factor != 0:
            h -= h % self.sr_factor
        if w % self.sr_factor != 0:
            w -= w % self.sr_factor
        # print(i, j, h ,w, end=' ')
        # print(i // self.sr_factor, j // self.sr_factor, h// self.sr_factor, w//self.sr_factor)
        # print(h, w)
        # h_lr, w_lr = h // self.sr_factor, w // self.sr_factor
        # if h % self.sr_factor != 0:
        #     h_lr = h // self.sr_factor - 1
        # else:
        #     h_lr = h // self.sr_factor
        # if w % self.sr_factor != 0:
        #     w_lr = w // self.sr_factor - 1
        # else:
        #     w_lr = w // self.sr_factor
        return F.crop(hr, i, j, h, w), F.crop(lr, i // self.sr_factor, j // self.sr_factor, h // self.sr_factor,
                                              w // self.sr_factor)


