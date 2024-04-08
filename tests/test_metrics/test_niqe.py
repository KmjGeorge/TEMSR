import cv2
import numpy as np
from tqdm import tqdm
from basicsr.metrics.niqe import calculate_niqe
import os


def test_niqe_folder(img_folder):
    niqe_avg = 0
    for filename in tqdm(os.listdir(img_folder)):
        img = cv2.imread(os.path.join(img_folder, filename), 0)
        img = img[..., np.newaxis]
        niqe = calculate_niqe(img, crop_border=0)
        # print('niqe =', niqe)
        niqe_avg += niqe
    niqe_avg /= len(os.listdir(img_folder))
    print('niqe_avg =', niqe_avg)
    return niqe_avg



if __name__ == '__main__':
    img_folder = 'F:\github\TEMSR\\results\SAFMN_FFT_STEM_test\\visualization\SimDeg'
    test_niqe_folder(img_folder)
