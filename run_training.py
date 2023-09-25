import torch
from torchsummary import summary
from utils.functions import setup_seed
from dataset.temimagenet import get_temimagenet_trainval
from models.SwinIR import get_swinir
from models.UHDFour import get_uhdfour
from models.DRAN import get_dran
from train.train import sr_train
from utils.functions import save_config
import configs


def get_model(name):
    if name == 'swinir':
        return get_swinir().to(configs.device)
    elif name == 'uhdfour':
        return get_uhdfour().to(configs.device)
    elif name == 'dran':
        return get_dran().to(configs.device)
    else:
        raise 'model not defined !'


if __name__ == '__main__':
    save_config(configs.training_config['task_name'])
    setup_seed(configs.training_config['seed'])

    model = get_model(configs.training_config['model'])
    summary(model, input_size=(configs.dataset_config['channel'], configs.multiscale_aug_config['orig_size'][0], configs.multiscale_aug_config['orig_size'][1]))
    # assert False
    train_loader, val_loader = get_temimagenet_trainval()
    sr_train(model, train_loader, val_loader, configs.training_config)

