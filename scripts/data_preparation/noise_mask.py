import pywt
import cv2
import os
import matplotlib.pyplot as plt
import tiffile
import statsmodels.api as sm
import torch
import numpy as np
from STEMdeg import grayimg2tensor, tensor2inp
from STEMdeg import random_add_poisson_noise_pt, random_add_gaussian_noise_pt, filter2D


def generate_mask(img, maxval):
    mean, std = img.mean(), img.std()
    thresh = mean + std * 0.7
    if thresh >= 0.8 * maxval:
        thresh = 0.8 * maxval
    print(mean, std, thresh)
    _, mask = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)
    mask = np.clip(mask + np.random.uniform(0.3, 0.8, mask.shape), 0, 1)
    mask = cv2.GaussianBlur(mask, (9, 9), 1)
    return mask


def add_spatially_varying_noise(img, noise_img, mask):
    return noise_img * mask+ img * (1 - mask)


if __name__ == '__main__':
    img = cv2.imread('G:\datasets\Sim ReSe2\\all\\0.3_0.0_0_0_35.0_0.6_1728.png', 0).astype(float) / 255.
    img_pt = grayimg2tensor(img)
    out = img_pt
    out, noise1 = random_add_poisson_noise_pt(out, scale_range=[3, 3], gray_prob=1)  # poisson noise
    out, noise2 = random_add_gaussian_noise_pt(
        out, sigma_range=[3, 3], gray_prob=1, scale_range=[8, 8])
    noisy_img1 = out[:, 0, :, :].unsqueeze(0)

    mask = generate_mask(img, 1)
    cv2.imshow('mask', mask)
    noisy_img2 = add_spatially_varying_noise(img_pt, noisy_img1, mask)

    img_noise1 = tensor2inp(noisy_img1)
    img_noise2 = tensor2inp(noisy_img2)
    cv2.imshow('noise1', img_noise1)
    cv2.imshow('noise2', img_noise2)
    cv2.waitKey(0)


