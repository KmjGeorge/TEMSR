import torch.nn as nn
import torch
import numpy as np
import random

dataset_config = {
    'path': 'G:/Datasets/TEM-ImageNet-v1.3-master/image/',
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
    'task_name': 'Test1',
    'seed': 50,
    'epochs': 200,
    'start_epoch': 0,
    'criterion': nn.CrossEntropyLoss(reduction='mean'),
    'optim_config': {
        'name': torch.optim.Adam,
        'lr': 5e-4,
        'weight_decay': 0.001,
    },
    'scheduler_config': {
        'multiplier': 1,
        'warmup_epoch': 15,
        'down_epoch': 100,
        'eta_min': 1e-2,
    },
    'validate_step': 1,
    'save': {
        'enable': True,
        'step': 1,
    }

}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(training_config['seed'])
