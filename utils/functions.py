import os
import random
import numpy as np
import torch
import configs
import json
from numpy import fft


def channel_swap(image):  # (c, h, w) -> (h, w, c)
    image = np.swapaxes(image, 0, 1)
    image = np.swapaxes(image, 1, 2)
    return image


def channel_expand(image):  # (1, h, w) -> (3, h, w)
    if len(image.shape) == 2:
        image = image[np.newaxis, ...]
    image = np.concatenate((image, image, image), axis=0)
    return image


def image_for_network(image, target_c=3):  # Single PIL image Object -> tensor for network input
    image_np = np.array(image)
    if len(image_np.shape) == 2:
        if target_c != image_np.shape[0]:
            image_np = channel_expand(image_np)
        if target_c == 1:
            image_np = image_np[np.newaxis, ...]
    image_tensor = (torch.from_numpy(image_np).float() / 255.0).unsqueeze_(0).to(configs.device)

    return image_tensor


def image_for_draw(image):  # network output tensor -> np array for plt drawing (h, w, c)
    return channel_swap(torch.squeeze(torch.squeeze(image)).cpu().detach().numpy())


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_config(savename):
    if not os.path.exists("./logs/{}".format(savename)):
        os.mkdir("./logs/{}".format(savename))
    config_all = [configs.dataset_config,
                  configs.multiscale_aug_config,
                  configs.training_config,
                  configs.swinir_config,
                  configs.uhdfour_config]
    with open('logs/{}/{}_config.json'.format(savename, savename), 'w') as f:
        json.dump(config_all, f)

