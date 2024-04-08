import cv2
import numpy as np
from tqdm import tqdm
from basicsr.metrics.niqe import calculate_niqe
import os
import cv2
import pywt
import numpy as np

from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_snr(img):
    # 应用Haar小波变换
    coeffs = pywt.dwt2(img, 'haar')
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
