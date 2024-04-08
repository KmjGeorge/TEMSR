import cv2
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
import tiffile
import statsmodels.api as sm

def show_statics(image_path1, image_path2):
    if '.tif' in image_path1:
        image1 = tiffile.imread(image_path1)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    else:
        image1 = cv2.imread(image_path1, 0)
    if 'tif' in image_path2:
        image2 = tiffile.imread(image_path2)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    else:
        image2 = cv2.imread(image_path2, 0)
    # 计算直方图
    hist1, bins1 = np.histogram(image1.flatten(), bins=256, range=(0, 256), density=True)
    hist2, bins2 = np.histogram(image2.flatten(), bins=256, range=(0, 256), density=True)
    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.bar(bins1[:-1], hist1, width=1, align='edge', color='blue', edgecolor='black')
    plt.title('Grayscale Image Histogram for Image1')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Number of Pixels')
    plt.xlim(0, 256)
    plt.show()

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.bar(bins2[:-1], hist2, width=1, align='edge', color='blue', edgecolor='black')
    plt.title('Grayscale Image Histogram for Image2')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Number of Pixels')
    plt.xlim(0, 256)
    plt.show()

    # 绘制QQ图
    fig = sm.qqplot_2samples(image1.flatten(), image2.flatten(), line='45')
    plt.title('Q-Q Plot of RealLQ and SimLQ')
    plt.savefig('Q-Q.png', dpi=200)
    plt.show()


if __name__ == '__main__':
    image_path1 = 'F:\Datasets\Sim ReSe2\simlq\\0.3_0.0_0_0_35.0_0.6_1728_deg0.png'
    image_path2 = 'F:\Datasets\STEM ReSe2\ReSe2\\all_GT\\20230829 2015 7.70 Mx 13 nm 0001 HAADF_s010.png'
    show_statics(image_path1, image_path2)
