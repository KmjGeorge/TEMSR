import os
import tiffile
import cv2
<<<<<<< HEAD
import os
path=  'F:\Datasets\STEM ReSe2\ReSe2\paired\offset\LQ_crops 256 256 png'
save_path = 'F:\Datasets\STEM ReSe2\ReSe2\paired\offset\LQ_crops 256 256 png bic64'
for filename in os.listdir(path):
    img = Image.open(os.path.join(path, filename)).convert('L').resize((64, 64), Image.BICUBIC)
    img.save(os.path.join(save_path, filename))




=======
path = 'G:\datasets\STEM ReSe2\ReSe2\\all_GT'
save_path = 'G:\datasets\STEM ReSe2\ReSe2\\all_GT png'
for filename in os.listdir(path):
    img_tif = tiffile.imread(os.path.join(path, filename))
    cv2.imwrite(os.path.join(save_path, filename.replace('.tif', '.png')), img_tif)
>>>>>>> 6e778a4948d5fe98403f37972edfc68781d5e3bf
