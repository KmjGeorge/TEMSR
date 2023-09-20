import torch.nn as nn
import torch
import numpy as np
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_config = {
    'lr_path': 'D:/Datasets/TEM-ImageNet-v1.3-master/image/',
    'hr_path': 'D:/Datasets/TEM-ImageNet-v1.3-master/noBackgroundnoNoise/',
    'channel': 3,
    'train_split': 0.8,
    'num_workers': 1,
    'batchsize': 16,
    'shuffle': True,
}

multiscale_aug_config = {
    'method': 'sub1',
    'orig_size': (256, 256),
    'crop_size': (128, 128),
    'scale': 0.5,
    'crop_scale': 0.5,
    'crop_times': 4
}

training_config = {
    'task_name': 'denosing+debg_uhdfour_3loss',
    'seed': 50,
    'epochs': 30,
    'start_epoch': 0,
    'model': 'uhdfour',
    'half_precision': False,
    'optim_config': {
        'name': 'Adam',
        'lr': 1e-5,
        'weight_decay': 0.001,
    },
    'scheduler_config': {
        'multiplier': 1,
        'warmup_epoch': 5,
        'eta_min': 1e-6,
    },
    'validation': {
        'enable': True,
        'step': 1
    },
    'save': {
        'enable': True,
        'step': 1,
    }

}

swinir_config = {
    'upscale': 1,
    'img_size': (256, 256),
    'window_size': 8,
    'img_range': 1.,
    'in_chans': 1,
    'depths': [2, 3, 3, 2],
    'embed_dim': 60,
    'num_heads': [4, 4, 4, 4],
    'mlp_ratio': 2,
    'upsampler': 'denoising'

}

uhdfour_config = {
    'in_chans': 3,
    'nc': 16
}



