import torch.nn as nn
import torch
import numpy as np
import random

dataset_config = {
    'path': 'D:/Datasets/TEM-ImageNet-v1.3-master/image/',
    'train_split': 0.8,
    'batchsize': 32,
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
    'task_name': 'Test1',
    'seed': 50,
    'epoch': 200,
    'start_epoch': 0,
    'criterion': nn.CrossEntropyLoss(reduction='mean'),
    'optim_config': {
        'name': torch.optim.Adam,
        'lr': 1e-3,
        'weight_decay': 0.001,
    },
    'scheduler_warmup_config': {
        'multiplier': 1,
        'total_epoch': 30,
    },
    'scheduler_down_config': {
        'total_epoch': 400,
        'eta_min': 1e-2,
    },

}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(training_config['seed'])
