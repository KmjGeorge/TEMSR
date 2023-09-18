import torch.nn as nn
import torch
import numpy as np
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_config = {
    'path': 'D:/Datasets/TEM-ImageNet-v1.3-master/image/',
    'train_split': 0.8,
    'num_workers': 1,
    'batchsize': 4,
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
    'task_name': 'swinIR_test1',
    'seed': 50,
    'epochs': 100,
    'start_epoch': 0,
    'criterion': 'ce',
    'model': 'swinIR',
    'optim_config': {
        'name': 'Adam',
        'lr': 5e-4,
        'weight_decay': 0.001,
    },
    'scheduler_config': {
        'multiplier': 1,
        'warmup_epoch': 5,
        'down_epoch': 60,
        'eta_min': 1e-2,
    },
    'validate_step': 1,
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
    'depths': [2, 2, 2, 2],
    'embed_dim': 60,
    'num_heads': [4, 4, 4, 4],
    'mlp_ratio': 2,
    'upsampler': 'denoising'

}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_config():
    import json
    configs = [dataset_config, multiscale_aug_config, training_config, swinir_config]
    with open('logs/{}.json'.format(training_config['task_name']), 'w') as f:
        json.dump(configs, f)


setup_seed(training_config['seed'])
