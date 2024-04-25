import pywt
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tiffile
import statsmodels.api as sm



path = 'G:\datasets\Sim ReSe2\\all\\wavelet'
# img = tiffile.imread(os.path.join(path.replace('\\wavelet', ''), '20230829 2237 7.70 Mx 13 nm B DCFI(HAADF).tif'))
# img = img[:, :, 0]
img = cv2.imread(os.path.join(path.replace('\\wavelet',''), '1.5_0.4_0_0_35.0_0.6_1704.png'), 0)
cv2.imwrite(os.path.join(path, '1.5_0.4_0_0_35.0_0.6_1704.png'), img)
cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')

cv2.imwrite(os.path.join(path, 'cA.png'), cA)
cv2.imwrite(os.path.join(path, 'cH.png'), cH +255)
cv2.imwrite(os.path.join(path, 'cV.png'), cV + 255)
cv2.imwrite(os.path.join(path, 'cD.png'), cD + 255)
high = pywt.idwt2((np.zeros_like(cA), (cH, cV, cD)), 'haar')
low = pywt.idwt2((cA, (np.zeros_like(cH), np.zeros_like(cH), np.zeros_like(cH))), 'haar')
cv2.imwrite(os.path.join(path, 'low.png'), low)
cv2.imwrite(os.path.join(path, 'high.png'), high)
compose = pywt.idwt2((cA, (cH, cV, np.zeros_like(cD))), 'haar')
cv2.imwrite(os.path.join(path, 'cA cH cV.png'), compose)

# 计算直方图
hist1, bins1 = np.histogram(cA.flatten(), bins=256, range=(0, 256), density=True)
hist2, bins2 = np.histogram(cH.flatten(), bins=256, range=(0, 256), density=True)
hist3, bins3 = np.histogram(cV.flatten(), bins=256, range=(0, 256), density=True)
hist4, bins4 = np.histogram(cD.flatten(), bins=256, range=(0, 256), density=True)
hist5, bins5 = np.histogram(low.flatten(), bins=256, range=(0, 256), density=True)
hist6, bins6 = np.histogram(high.flatten(), bins=256, range=(0, 256), density=True)

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.bar(bins1[:-1], hist1, width=1, align='edge', color='blue', edgecolor='black')
plt.title('Grayscale Image Histogram for cA')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
plt.xlim(0, 256)
plt.savefig(os.path.join(path, 'cA_histogram.png'))
plt.show()

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.bar(bins2[:-1], hist2, width=1, align='edge', color='blue', edgecolor='black')
plt.title('Grayscale Image Histogram for cH')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
plt.xlim(0, 256)
plt.savefig(os.path.join(path, 'cH_histogram.png'))
plt.show()

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.bar(bins3[:-1], hist3, width=1, align='edge', color='blue', edgecolor='black')
plt.title('Grayscale Image Histogram for cV')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
plt.xlim(0, 256)
plt.savefig(os.path.join(path, 'cV_histogram.png'))
plt.show()

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.bar(bins3[:-1], hist4, width=1, align='edge', color='blue', edgecolor='black')
plt.title('Grayscale Image Histogram for cD')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
plt.xlim(0, 256)
plt.savefig(os.path.join(path, 'cD_histogram.png'))
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(bins5[:-1], hist5, width=1, align='edge', color='blue', edgecolor='black')
plt.title('Grayscale Image Histogram for low')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
plt.xlim(0, 256)
plt.savefig(os.path.join(path, 'low_histogram.png'))
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(bins6[:-1], hist6, width=1, align='edge', color='blue', edgecolor='black')
plt.title('Grayscale Image Histogram for high')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
plt.xlim(0, 256)
plt.savefig(os.path.join(path, 'high_histogram.png'))
plt.show()

