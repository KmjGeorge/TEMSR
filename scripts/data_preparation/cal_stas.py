import cv2
from PIL import Image
import tiffile
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from tqdm import tqdm

def z_score(image):
    mean = image.mean()
    std = image.std()
    return (image - mean) / std

def adaptive_histogram_equalization(image, tile_grid_size=(8, 8)):
    """
    对图像进行自适应直方图均衡化。

    参数:
    - image: 输入的灰度图像。
    - tile_grid_size: 直方图均衡化的栅格尺寸，例如 (8, 8) 表示图像将被分割成 8x8 大小的区域。

    返回:
    - 输出图像，经过自适应直方图均衡化处理。
    """
    # 确保输入图像是灰度的
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 获取图像尺寸
    img_height, img_width = image.shape

    # 计算每个tile的尺寸
    tile_width = img_width // tile_grid_size[1]
    tile_height = img_height // tile_grid_size[0]

    # 初始化输出图像
    output_image = np.zeros_like(image)

    # 遍历每个tile
    for i in range(tile_grid_size[0]):
        for j in range(tile_grid_size[1]):
            # 计算当前tile的边界
            y1, y2 = i * tile_height, (i + 1) * tile_height
            x1, x2 = j * tile_width, (j + 1) * tile_width

            # 提取当前tile
            tile = image[y1:y2, x1:x2]

            # 计算当前tile的直方图
            hist, _ = np.histogram(tile, bins=256, range=(0, 256))

            # 计算累积分布函数（CDF）
            cdf = np.cumsum(hist) * (256 / (256 * len(hist)))

            # 计算均衡化映射
            eq_map = (255 / cdf[-1]) * cdf

            # 将映射应用到当前tile
            output_image[y1:y2, x1:x2] = eq_map[tile]

    return output_image


def histogram_equalization(image):
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算灰度级的分布
    hist, bins = np.histogram(gray_image.flatten(), 256, [0, 256])
    # 计算累积分布函数(cumulative distribution function, CDF)
    cdf = hist.cumsum()
    # 归一化cdf到[0,255]的整数范围
    cdf_normalized = 255 * cdf / cdf.max()
    # 将CDF函数映射到新的灰度级
    gray_image_equalized = np.array(cdf_normalized[gray_image], dtype=np.uint8)
    # 将图像转换回原来的颜色空间
    equalized_image = cv2.cvtColor(gray_image_equalized, cv2.COLOR_GRAY2BGR)
    return equalized_image


def histogram_clahe(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
    gray_image_equalized = clahe.apply(gray_image)
    equalized_image = cv2.cvtColor(gray_image_equalized, cv2.COLOR_GRAY2BGR)
    return equalized_image


def calucale_mean_std(path):
    mean = 0
    var = 0
    length = 0
    folder = ['MoS2', 'ReS2', 'ReSe2', 'warwick']
    for i in range(4):
        length += len(os.listdir(os.path.join(path, folder[i])))
        for filename in tqdm(os.listdir(os.path.join(path, folder[i]))):
            if '.tif' in filename:
                img = tiffile.imread(os.path.join(path, folder[i], filename))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
            else:
                img = cv2.imread(os.path.join(path, folder[i], filename), 0) / 255.0
            mean += img.mean()
            var += img.var()
    mean /= length
    var /= length
    std = np.sqrt(var)
    return mean, std


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
    path = 'F:\\Datasets\\STEMEXP'
    mean, std = calucale_mean_std(path)
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
