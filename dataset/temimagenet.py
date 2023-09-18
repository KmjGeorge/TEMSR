import os

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import h5py
import random
import numpy as np
import dataset.multiscale_aug as multiscale_aug
import configs
from tqdm import tqdm


class TEMImageNet_Aug(Dataset):
    def __init__(self, path, aug):
        orig_images = []
        self.lrs = []
        self.hrs = []

        for filename in os.listdir(path):
            image = Image.open(os.path.join(path, filename)).resize(aug['orig_size'], Image.Resampling.LANCZOS).convert(
                'L')
            orig_images.append(image)

        if aug['method']:
            self.hrs = orig_images
            for image in orig_images:
                resize = image.resize((aug['downscale'] * aug['orig_size'][0], aug['downscale'] * aug['orig_size'][1]),
                                      Image.NEAREST)
                resize = np.array(resize)
                self.lrs.append(resize)

        elif aug['method'] == 'sub1':
            self.lrs, self.hrs = multiscale_aug.sub_process1(images=orig_images,
                                                             hr_size=aug['crop_size'],
                                                             crop_down_scale=aug['crop_scale'],
                                                             crop_times=aug['crop_times'],
                                                             )
        elif aug['method'] == 'sub2':
            self.lrs, self.hrs = multiscale_aug.sub_process2(images=orig_images,
                                                             hr_size=aug['crop_size'],
                                                             down_scale=aug['scale'],
                                                             crop_down_scale=aug['crop_scale'],
                                                             crop_times=aug['crop_times'],
                                                             )
        elif aug['method'] == 'sub3':
            self.lrs, self.hrs = multiscale_aug.sub_process3(images=orig_images,
                                                             hr_size=aug['crop_size'],
                                                             up_scale=aug['scale'],
                                                             crop_down_scale=aug['crop_scale'],
                                                             crop_times=aug['crop_times'],
                                                            )
        elif aug['method'] == 'sub4':
            self.lrs, self.hrs = multiscale_aug.sub_process4(images=orig_images,
                                                             lr_size=aug['crop_size'],
                                                             down_scale=aug['scale'],
                                                             crop_up_scale=aug['crop_scale'],
                                                             crop_times=aug['crop_times'],
                                                             )

    def __len__(self):
        return len(self.lrs)

    def __getitem__(self, idx):
        return self.lrs[idx], self.hrs[idx]


class TEMImageNet(Dataset):
    def __init__(self, path, hr_path, aug):
        self.hrs = []
        self.lrs = []
        self.filenames = []
        for filename in tqdm(os.listdir(path)):
            image = Image.open(os.path.join(path, filename)).resize(aug['orig_size'], Image.Resampling.LANCZOS).convert(
                'L')
            image = np.array(image)
            image = image[np.newaxis, ...]
            self.lrs.append(image)
            self.filenames.append(filename)
        for filename in tqdm(os.listdir(hr_path)):
            image = Image.open(os.path.join(hr_path, filename)).convert('L')
            image = np.array(image)
            image = image[np.newaxis, ...]
            self.hrs.append(image)

    def __len__(self):
        return len(self.lrs)

    def __getitem__(self, idx):
        return self.lrs[idx], self.hrs[idx], self.filenames[idx]


def get_temimagenet_trainval():
    dataset = TEMImageNet(path=configs.dataset_config['path'],
                          hr_path=configs.dataset_config['path'] + '../noNoise/',
                          aug=configs.multiscale_aug_config)
    train_split = configs.dataset_config['train_split']
    train_dataset, val_dataset = random_split(dataset, [train_split, 1 - train_split])
    print('train_length =', len(train_dataset), 'val_length =', len(val_dataset))
    train = DataLoader(train_dataset, batch_size=configs.dataset_config['batchsize'],
                       shuffle=configs.dataset_config['shuffle'], num_workers=configs.dataset_config['num_workers'])
    val = DataLoader(val_dataset, batch_size=configs.dataset_config['batchsize'],
                     shuffle=configs.dataset_config['shuffle'], num_workers=configs.dataset_config['num_workers'])
    return train, val


# def get_temimagenet_test():
#     test_dataset = TEMImageNet(path=configs.dataset_config['path'], aug=configs.multiscale_aug_config)
#     test = DataLoader(test_dataset, batch_size=configs.dataset_config['batchsize'],
#                       shuffle=configs.dataset_config['shuffle'])
#     return test

if __name__ == '__main__':
    train, val = get_temimagenet_trainval()

    i = 1
    for lr, hr, filename in train:
        if i == 2:
            break
        for j in range(4):
            plt.subplot(121)
            plt.title('{} lr'.format(filename[j]))
            plt.imshow(np.squeeze(lr[j]), 'gray')
            plt.subplot(122)
            plt.title('{} hr'.format(filename[j]))
            plt.imshow(np.squeeze(hr[j]), 'gray')
            plt.show()
            i += 1
