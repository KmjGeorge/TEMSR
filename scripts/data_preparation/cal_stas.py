import cv2
from PIL import Image
import tiffile
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from tqdm import tqdm

def calucale_mean_std(path):
    mean = 0
    var = 0
    length = 0
    length += len(os.listdir(path))
    for filename in tqdm(os.listdir(path)):
        if '.tif' in filename:
            img = tiffile.imread(os.path.join(path, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
        else:
            img = cv2.imread(os.path.join(path, filename), 0) / 255.0
        mean += img.mean()
        var += img.var()
    mean /= length
    var /= length
    std = np.sqrt(var)
    return mean, std, var


def draw_avg_histogram(path, title, save_fig_path=None):
    hist_acc = 0
    length = len(os.listdir(path))
    for filename in tqdm(os.listdir(path)):
        if '.tif' in filename:
            img = tiffile.imread(os.path.join(path, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.imread(os.path.join(path, filename), 0)
        hist, _ = np.histogram(img.flatten(), 256, range=(0, 256))
        hist_acc += hist
    avg_hist = hist_acc / length
    plt.figure(figsize=(10, 5))
    plt.plot(avg_hist, color='blue', linestyle='-', linewidth=2)
    plt.title('Average Grayscale Histogram for {}'.format(title))
    plt.xlabel('Pixel intensity')
    plt.ylabel('Average number of pixels')
    plt.xlim([0, 256])
    if save_fig_path:
        plt.savefig(save_fig_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    path1 = r'F:\Datasets\InstructSTEMIR\Denoise\GT\TEMImageNet2kExp2k'
    path2 = r'F:\Datasets\InstructSTEMIR\Denoise\LQ\TEMImageNet2kExp2k'
    mean1, _, var1 = calucale_mean_std(path1)
    mean2, _, var2 = calucale_mean_std(path2)
    mean = (mean1 + mean2) / 2
    std = np.sqrt((var1 + var2) / 2)
    print('mean=', mean, 'std=', std)
    # draw_avg_histogram(path, 'Atom Crops', 'F:\Datasets\partial-STEM_full_size\\atom_crop_hist.png')


    '''
    image = tiffile.imread('G:\datasets\STEM ReSe2\ReSe2\paired\offset\GT\\2219_GT_x19y2.tif')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    draw_histogram(image)
    image_norm = z_score(image)
    image_out = histogram_equalization(image)
    image_out_ad = histogram_clahe(image)
    draw_histogram(image_out)
    draw_histogram(image_out_ad)
    tiffile.imwrite('G:\datasets\STEM ReSe2\ReSe2\paired\offset\GT\\2219_GT_x19y2_eq.tif', image_out)
    tiffile.imwrite('G:\datasets\STEM ReSe2\ReSe2\paired\offset\GT\\2219_GT_x19y2_eqad.tif', image_out_ad)
    '''
