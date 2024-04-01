import os.path
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import cv2


def draw_qq(path1, path2):
    img1 = cv2.imread(path1, 0)
    img2 = cv2.imread(path2, 0)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止中文标签乱码
    plt.rcParams['axes.unicode_minus'] = False
    data = img1.reshape(1, img1.shape[0]*img1.shape[1]).squeeze(0)

    plt.figure(figsize=(6, 6))
    # 绘制概率图（probability plot）
    # stats.probplot函数通过最小二乘法来估计一组数据的分位数对，并利用线性回归技术求出分位数图上的理论值与实际值的直线方程。
    stats.probplot(data, plot=plt, dist='norm', fit=True, rvalue=True)
    plt.title('Probability Plot (Q-Q Plot)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # 显示图形
    plt.show()


draw_qq('D:\Datasets\STEM ReSe2\ReSe2\paired\offset\\LQ_crops 256 256 png\\2219_x19y2_s001.png',
        'D:\Datasets\STEM ReSe2\ReSe2\paired\offset\\LQ_crops(deg from GT) 256 256 png\\2219_x19y2_s001.png')
