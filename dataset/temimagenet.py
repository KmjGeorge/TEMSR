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
from utils.functions import channel_swap, channel_expand

# TODO
class TEMImageNet_Aug(Dataset):
    def __init__(self, path, aug_config, channel=1):
        orig_images = []
        self.lrs = []
        self.hrs = []

        for filename in os.listdir(path):
            image = Image.open(os.path.join(path, filename)).resize(aug_config['orig_size'], Image.Resampling.LANCZOS)
            orig_images.append(image)

        if aug_config['method']:
            self.hrs = orig_images
            for image in orig_images:
                resize = image.resize((aug_config['downscale'] * aug_config['orig_size'][0],
                                       aug_config['downscale'] * aug_config['orig_size'][1]),
                                      Image.NEAREST)
                resize = np.array(resize)
                self.lrs.append(resize)

        elif aug_config['method'] == 'sub1':
            self.lrs, self.hrs = multiscale_aug.sub_process1(images=orig_images,
                                                             hr_size=aug_config['crop_size'],
                                                             crop_down_scale=aug_config['crop_scale'],
                                                             crop_times=aug_config['crop_times'],
                                                             )
        elif aug_config['method'] == 'sub2':
            self.lrs, self.hrs = multiscale_aug.sub_process2(images=orig_images,
                                                             hr_size=aug_config['crop_size'],
                                                             down_scale=aug_config['scale'],
                                                             crop_down_scale=aug_config['crop_scale'],
                                                             crop_times=aug_config['crop_times'],
                                                             )
        elif aug_config['method'] == 'sub3':
            self.lrs, self.hrs = multiscale_aug.sub_process3(images=orig_images,
                                                             hr_size=aug_config['crop_size'],
                                                             up_scale=aug_config['scale'],
                                                             crop_down_scale=aug_config['crop_scale'],
                                                             crop_times=aug_config['crop_times'],
                                                             )
        elif aug_config['method'] == 'sub4':
            self.lrs, self.hrs = multiscale_aug.sub_process4(images=orig_images,
                                                             lr_size=aug_config['crop_size'],
                                                             down_scale=aug_config['scale'],
                                                             crop_up_scale=aug_config['crop_scale'],
                                                             crop_times=aug_config['crop_times'],
                                                             )

    def __len__(self):
        return len(self.lrs)

    def __getitem__(self, idx):
        return self.lrs[idx], self.hrs[idx]


class TEMImageNet(Dataset):
    def __init__(self, lr_path, hr_path, aug_config, channel=1):
        self.lr_files = []
        self.hr_files = []
        self.channel = channel
        self.lr_size = aug_config['orig_size']
        for filename in os.listdir(lr_path):
            self.lr_files.append(os.path.join(lr_path, filename))

        for filename in os.listdir(hr_path):
            self.hr_files.append(os.path.join(hr_path, filename))

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_image = Image.open(self.lr_files[idx]).resize(self.lr_size, Image.Resampling.LANCZOS)
        lr_image = np.array(lr_image)
        lr_image = lr_image[np.newaxis, ...]
        if self.channel == 3:
            lr_image = channel_expand(lr_image)

        hr_image = Image.open(self.hr_files[idx])
        hr_image = np.array(hr_image)
        hr_image = hr_image[np.newaxis, ...]
        if self.channel == 3:
            hr_image = channel_expand(hr_image)

        return lr_image, hr_image, self.lr_files[idx], self.hr_files[idx]


def down_interporlation(path, save_path, target_size=(128, 128), method=Image.BICUBIC):
    for filename in tqdm(os.listdir(path)):
        img = Image.open(os.path.join(path, filename)).resize(target_size, Image.BICUBIC)
        img.save(os.path.join(save_path, filename))



def get_temimagenet_trainval():
    dataset = TEMImageNet(lr_path=configs.dataset_config['lr_path'],
                          hr_path=configs.dataset_config['hr_path'],
                          aug_config=configs.multiscale_aug_config,
                          channel=configs.dataset_config['channel'])
    train_split = configs.dataset_config['train_split']
    train_dataset, val_dataset = random_split(dataset, [train_split, 1 - train_split],
                                              generator=torch.Generator().manual_seed(configs.training_config['seed']))
    print('train_length =', len(train_dataset), 'val_length =', len(val_dataset))
    train = DataLoader(train_dataset, batch_size=configs.dataset_config['batchsize'],
                       shuffle=configs.dataset_config['shuffle'], num_workers=configs.dataset_config['num_workers'])
    val = DataLoader(val_dataset, batch_size=configs.dataset_config['batchsize'],
                     shuffle=configs.dataset_config['shuffle'], num_workers=configs.dataset_config['num_workers'])
    return train, val


if __name__ == '__main__':

    train, val = get_temimagenet_trainval()
    from utils.functions import setup_seed
    setup_seed(50)
    i = 1
    for lr, hr, file_lr, file_hr in train:
        lr = lr.cuda().float()
        hr = hr.cuda().float()
        # if i == 2:
        #     break
        # for j in range(2):
        #     plt.subplot(121)
        #     filename_lr = os.path.split(file_lr[j])[1]
        #     plt.title('{} lr'.format(filename_lr))
        #     plt.imshow(channel_swap(np.squeeze(lr[j])))
        #     plt.subplot(122)
        #     filename_hr = os.path.split(file_hr[j])[1]
        #     plt.title('{} hr'.format(filename_hr))
        #     plt.imshow(channel_swap(np.squeeze(hr[j])))
        #     plt.show()
        # i += 1

