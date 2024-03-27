import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from basicsr.utils import scandir


def get_down2x(path, save_path):
    for filename in tqdm(os.listdir(path)):
        img = Image.open(os.path.join(path, filename)).convert('L')
        img_down = img.resize((128, 128), Image.Resampling.BICUBIC)
        img_down.save(os.path.join(save_path, filename))


def calcuate_mean(path, size):
    img_list = os.listdir(path)
    img_example = Image.open(os.path.join(path, img_list[0]))
    img_size = size
    mode = img_example.mode
    # print(mode)
    if mode == 'L':
        batch_img = np.zeros([len(img_list), img_size[1], img_size[0]], dtype=np.uint8)
    elif mode == 'RGB':
        batch_img = np.zeros([len(img_list), img_size[1], img_size[0], 3], dtype=np.uint8)
    else:
        raise 'Error Image Mode'
    for i, filename in tqdm(enumerate(img_list)):
        img = Image.open(os.path.join(path, filename)).resize(size)
        img = np.array(img)
        batch_img[i] = img
    batch_img = batch_img / 255.0
    mean = batch_img.mean(axis=(0, 1, 2))
    return mean


def get_metainfo():
    gt_folder = 'D:/Datasets/TEM-ImageNet-v1.3-master/noBackgroundnoNoise/train'
    meta_info_txt = 'D:/github/TEMSR_BasicSR/basicsr/data/meta_info/meta_info_TEMImageNet_GT.txt'

    img_list = sorted(list(scandir(gt_folder)))

    with open(meta_info_txt, 'w') as f:
        for idx, img_path in enumerate(img_list):
            img = Image.open(os.path.join(gt_folder, img_path))  # lazy load
            width, height = img.size
            mode = img.mode
            if mode == 'RGB':
                n_channel = 3
            elif mode == 'L':
                n_channel = 1
            else:
                raise ValueError(f'Unsupported mode {mode}.')

            info = f'{img_path} ({height},{width},{n_channel})'
            print(idx + 1, info)
            f.write(f'{info}\n')


if __name__ == '__main__':
    # get_metainfo()
    path = 'D:\Datasets\TEMPatch for SR\GT\\full'
    print(calcuate_mean(path, (256, 256)))  # temimagenet 0.19789395157031994      # temsr  0.60797698

