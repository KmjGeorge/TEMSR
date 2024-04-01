import numpy as np
import tiffile
import cv2

img = cv2.imread('D:\Datasets\STEM ReSe2\ReSe2\singlelayer\GT\crops\\20230829 2015 7.70 Mx 13 nm B DCFI(HAADF)_s007.png', 0)
ret, thresh_img = cv2.threshold(img, thresh=150, maxval=255, type=cv2.THRESH_BINARY)
cv2.imwrite('D:\Datasets\STEM ReSe2\ReSe2\singlelayer\GT\\20230829 2015 7.70 Mx 13 nm B DCFI(HAADF)_s007_thresh.png', thresh_img)