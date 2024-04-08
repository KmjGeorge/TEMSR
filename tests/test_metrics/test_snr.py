import cv2
import numpy as np
from tqdm import tqdm
from basicsr.metrics.niqe import calculate_niqe
import os
import cv2
import pywt
import numpy as np


def calculate_snr(image):
    # 应用Haar小波变换
    coeffs = pywt.dwt2(image, 'haar')
    # 提取低频系数（LL3, LL2, LL1）
    LL3, (LH, HL, HH) = coeffs
    # 信号（低频部分）
    signal = LL3

    # 重构噪声（使用非低频系数）
    noise = np.zeros_like(LL3)
    noise = pywt.idwt2((noise, (LH, HL, HH)), 'haar')
    # 计算信号和噪声的功率
    signal_power = np.sum(signal ** 2) / (signal.size)
    noise_power = np.sum(noise ** 2) / (noise.size)
    # 避免除以零
    if noise_power == 0:
        noise_power = 1e-10
    # 计算SNR
    snr = signal_power / noise_power

    # 将SNR转换为分贝
    snr_db = 10 * np.log10(snr)
    return snr_db


def test_snr_folder(img_folder):
    snr_avg = 0
    for filename in tqdm(os.listdir(img_folder)):
        img = cv2.imread(os.path.join(img_folder, filename), 0)
        img = img[..., np.newaxis]
        snr = calculate_snr(img)
        print('snr =', snr)
        snr_avg += snr
    snr_avg /= len(os.listdir(img_folder))
    print('snr_avg =', snr_avg)
    return snr_avg


if __name__ == '__main__':
    img_folder = 'F:\Datasets\Sim ReSe2\simval\GT'
    test_snr_folder(img_folder)
