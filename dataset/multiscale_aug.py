from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# def load(path, orig_size=(350, 350)):
#     images = []
#     for filename in os.listdir(path):
#         image = Image.open(os.path.join(path, filename)).resize(orig_size).convert('L')
#         images.append(image)
#     return images


def bicubic_up(images, scale):
    images_up = []
    shape = images[0].size
    if scale > 1.0:
        new_shape = (int(shape[0] * scale), int(shape[1] * scale))
        for image in images:
            image_up = image.resize(new_shape, Image.BICUBIC)
            images_up.append(image_up)
    else:
        raise 'Error UpSample Scale!'
    return images_up


def nearest_down(images, scale):
    images_down = []
    shape = images[0].size
    if 0 < scale < 1.0:
        new_shape = (int(shape[0] * scale), int(shape[1] * scale))
        for image in images:
            image_down = image.resize(new_shape, Image.NEAREST)
            images_down.append(image_down)
    else:
        raise 'Error DownSample Scale!'
    return images_down


def downscaling_set(image, scale):
    shape = image.size
    if 0 < scale < 1.0:
        new_shape = (int(shape[0] * scale), int(shape[1] * scale))
        image_down_nearest = image.resize(new_shape, Image.NEAREST)
        image_down_lanczos = image.resize(new_shape, Image.LANCZOS)
        image_down_bilinear = image.resize(new_shape, Image.BILINEAR)
        image_down_bicubic = image.resize(new_shape, Image.BICUBIC)
        image_down_box = image.resize(new_shape, Image.BOX)
        image_down_hamming = image.resize(new_shape, Image.HAMMING)
        return image_down_nearest, image_down_lanczos, image_down_bilinear, image_down_bicubic, image_down_box, image_down_hamming
    else:
        raise 'Error DownSample Scale!'


def sub_process1(images, hr_size, crop_down_scale, crop_times=3):
    crop_hrs = []
    crop_lrs = []
    for image in tqdm(images):
        for i in range(crop_times):
            randomcrop_pos = np.random.randint(0, image.size[0] - max(hr_size[0], hr_size[1]))
            crop_hr = images.crop(
                (randomcrop_pos, randomcrop_pos, randomcrop_pos + hr_size[0], randomcrop_pos + hr_size[1]))
            crop_lr_6v = downscaling_set(crop_hr, crop_down_scale)
            for crop_lr in crop_lr_6v:
                crop_hrs.append(np.array(crop_hr))
                crop_lrs.append(np.array(crop_lr))
    return crop_lrs, crop_hrs


def sub_process2(images, hr_size, down_scale, crop_down_scale, crop_times=3):
    crop_hrs = []
    crop_lrs = []
    images_down = nearest_down(images, down_scale)
    for image in tqdm(images_down):
        for i in range(crop_times):
            randomcrop_pos = np.random.randint(0, image.size[0] - max(hr_size[0], hr_size[1]))
            crop_hr = images.crop(
                (randomcrop_pos, randomcrop_pos, randomcrop_pos + hr_size[0], randomcrop_pos + hr_size[1]))
            crop_lr_6v = downscaling_set(crop_hr, crop_down_scale)
            for crop_lr in crop_lr_6v:
                crop_hrs.append(np.array(crop_hr))
                crop_lrs.append(np.array(crop_lr))
    return crop_lrs, crop_hrs


def sub_process3(images, hr_size, up_scale, crop_down_scale, crop_times=3):
    crop_hrs = []
    crop_lrs = []
    images_up = bicubic_up(images, up_scale)
    for image in tqdm(images_up):
        for i in range(crop_times):
            randomcrop_pos = np.random.randint(0, image.size[0] - max(hr_size[0], hr_size[1]))
            crop_hr = images.crop(
                (randomcrop_pos, randomcrop_pos, randomcrop_pos + hr_size[0], randomcrop_pos + hr_size[1]))
            crop_lr_6v = downscaling_set(crop_hr, crop_down_scale)
            for crop_lr in crop_lr_6v:
                crop_hrs.append(np.array(crop_hr))
                crop_lrs.append(np.array(crop_lr))
    return crop_lrs, crop_hrs


def sub_process4(images, lr_size, down_scale, crop_up_scale, crop_times=3):
    crop_hrs = []
    crop_lrs = []
    images_down = nearest_down(images, down_scale)
    for image in tqdm(images_down):
        for i in range(crop_times):
            randomcrop_pos = np.random.randint(0, image.size[0] - max(lr_size[0], lr_size[1]))
            crop_lr = images.crop(
                (randomcrop_pos, randomcrop_pos, randomcrop_pos + lr_size[0], randomcrop_pos + lr_size[1]))
            crop_hr = crop_lr.resize((int(lr_size[0] * crop_up_scale), int(lr_size[1] * crop_up_scale)), Image.BICUBIC)
            crop_lrs.append(np.array(crop_lr))
            crop_hrs.append(np.array(crop_hr))
    return crop_lrs, crop_hrs
