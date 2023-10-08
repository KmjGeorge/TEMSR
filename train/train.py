import os.path

import lpips
import matplotlib.pyplot as plt
import torch.cuda
from torch.optim.lr_scheduler import CosineAnnealingLR

import configs
from utils.scheduler import GradualWarmupScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
from tqdm import tqdm
from utils.metrics import cal_psnr, cal_ssim, PerceptualLoss
import torch.nn.functional as F
import numpy as np
from utils.optimizer import Lion
from utils.metrics import cal_atom_loss, atom_loss_fn
device = configs.device


class Log:
    def __init__(self):
        self.epoch_list = []
        self.loss_list = []
        self.psnr_list = []
        self.ssim_list = []
        self.lpips_list = []
        self.lr_list = []

    def add(self, epoch, loss, psnr, ssim, lpips, lr):
        self.epoch_list.append(epoch)
        self.loss_list.append(loss)
        self.psnr_list.append(psnr)
        self.ssim_list.append(ssim)
        self.lpips_list.append(lpips)
        self.lr_list.append(lr)

    def __str__(self):
        epoch_info = 'epoch: ' + str(self.epoch_list) + '\n'
        loss_info = 'loss: ' + str(self.loss_list) + '\n'
        psnr_info = 'psnr: ' + str(self.psnr_list) + '\n'
        ssim_info = 'ssim: ' + str(self.ssim_list) + '\n'
        lpips_info = 'lpips: ' + str(self.lpips_list) + '\n'
        lr_info = 'lr: ' + str(self.lr_list) + '\n'
        return epoch_info + loss_info + psnr_info + ssim_info + lpips_info + lr_info


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def sr_train(model, train_dataloader, val_dataloader, training_config):
    perception_fn = lpips.LPIPS(net='vgg').to(configs.device)
    if training_config['atom_loss']['enable']:
        atom_fn = atom_loss_fn(training_config['atom_loss']['unet_path'])
    else:
        atom_fn = None
    train_log = Log()
    val_log = Log()
    optim_config = training_config['optim_config']
    scheduler_config = training_config['scheduler_config']
    if optim_config['name'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=optim_config['lr'],
                                     weight_decay=optim_config['weight_decay'], betas=optim_config['betas'])
    elif optim_config['name'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=optim_config['lr'],
                                      weight_decay=optim_config['weight_decay'], betas=optim_config['betas'])
    elif optim_config['name'] == 'Lion':
        optimizer = Lion(model.parameters(), lr=optim_config['lr'],
                         weight_decay=optim_config['weight_decay'], betas=optim_config['betas'])
    else:
        raise 'Error optimizer name!'
    scheduler_cos = CosineAnnealingLR(optimizer,
                                      T_max=training_config['epochs'] - scheduler_config['warmup_epoch'],
                                      eta_min=scheduler_config['eta_min'])
    scheduler = GradualWarmupScheduler(optimizer, multiplier=scheduler_config['multiplier'],
                                       total_epoch=scheduler_config['warmup_epoch'],
                                       after_scheduler=scheduler_cos)
    scheduler.step()
    scaler = GradScaler()
    for epoch in range(1, training_config['epochs'] + 1):

        # train
        epoch_loss, epoch_psnr, epoch_ssim, epoch_lpips = train_epoch(model, train_dataloader, optimizer, perception_fn, atom_fn,
                                                                      scaler, epoch)
        train_log.add(epoch, epoch_loss, epoch_psnr, epoch_ssim, epoch_lpips, optimizer.param_groups[0]['lr'])

        # val
        if training_config['validation']['enable']:
            if epoch % training_config['validation']['step'] == 0:
                val_loss, val_psnr, val_ssim, val_lpips = validate_epoch(model, val_dataloader, epoch, perception_fn, atom_fn)
                val_log.add(epoch, val_loss, val_psnr, val_ssim, val_lpips, np.nan)
        # save
        if training_config['save']['enable']:
            if epoch % training_config['save']['step'] == 0:
                save(train_log, val_log, model, epoch=epoch, savename=training_config['task_name'],
                     start_epoch=training_config['start_epoch'])

        scheduler.step()


def train_epoch(model, train_dataloader, optimizer, perception_loss_fn, atom_fn, scaler, epoch):
    model.train()
    loop = tqdm(train_dataloader)
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    lpips_meter = AverageMeter()

    for lr, hr, _, _ in loop:
        lr = (lr.float() / 255.0).to(device)
        hr = (hr.float() / 255.0).to(device)

        optimizer.zero_grad()
        if configs.training_config['half_precision']:
            with autocast():
                output = model(lr).to(device)
                l1_loss = 5 * F.smooth_l1_loss(output, hr)
                ssim_loss = 1 * (1 - cal_ssim(output, hr))
                perception_loss = 1 * perception_loss_fn(output, hr)
                perception_loss = torch.mean(perception_loss, dim=0).squeeze_().squeeze_().squeeze_()
                if configs.training_config['atom_loss']['enable']:
                    atom_loss = configs.training_config['atom_loss']['loss_weight'] * cal_atom_loss(atom_fn, output, hr)
                    loss = l1_loss + ssim_loss + perception_loss + atom_loss
                else:
                    loss = l1_loss + ssim_loss + perception_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            output = model(lr).to(device)
            l1_loss = 5 * F.smooth_l1_loss(output, hr)
            ssim_loss = 1 * (1 - cal_ssim(output, hr))
            perception_loss = perception_loss_fn(output, hr)
            perception_loss = 1 * torch.mean(perception_loss, dim=0).squeeze_().squeeze_().squeeze_()
            if configs.training_config['atom_loss']['enable']:
                atom_loss = configs.training_config['atom_loss']['loss_weight'] * cal_atom_loss(atom_fn, output, hr)
                loss = l1_loss + ssim_loss + perception_loss + atom_loss
            else:
                loss = l1_loss + ssim_loss + perception_loss
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            loss_meter.update(loss.item())
            psnr_meter.update(cal_psnr(output, hr).item())
            ssim_meter.update(cal_ssim(output, hr).item())
            lpips_meter.update(perception_loss.item())

        loop.set_description('Epoch {}'.format(epoch+configs.training_config['start_epoch']))
        loop.set_postfix(loss=loss_meter.avg,
                         psnr=psnr_meter.avg,
                         ssim=ssim_meter.avg,
                         lpips=lpips_meter.avg,
                         lr=optimizer.param_groups[0]['lr'])
    epoch_loss = loss_meter.avg
    epoch_psnr = psnr_meter.avg
    epoch_ssim = ssim_meter.avg
    epoch_lpips = lpips_meter.avg
    return epoch_loss, epoch_psnr, epoch_ssim, epoch_lpips


def validate_epoch(model, val_dataloader, epoch, perception_loss_fn, atom_fn):
    model.eval()
    loop = tqdm(val_dataloader)
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    lpips_meter = AverageMeter()

    for lr, hr, _, _ in loop:
        lr = (lr.float() / 255.0).to(device)
        hr = (hr.float() / 255.0).to(device)
        with torch.no_grad():
            if configs.training_config['half_precision']:
                with autocast():
                    output = model(lr).to(device)
                    l1_loss = 5 * F.smooth_l1_loss(output, hr)
                    ssim_loss = 1 * (1 - cal_ssim(output, hr))
                    perception_loss = 1 * perception_loss_fn(output, hr)
                    perception_loss = torch.mean(perception_loss, dim=0).squeeze_().squeeze_().squeeze_()
                    if configs.training_config['atom_loss']['enable']:
                        atom_loss = configs.training_config['atom_loss']['loss_weight'] * cal_atom_loss(atom_fn, output, hr)
                        loss = l1_loss + ssim_loss + perception_loss + atom_loss
                    else:
                        loss = l1_loss + ssim_loss + perception_loss
            else:
                output = model(lr).to(device)
                l1_loss = 5 * F.smooth_l1_loss(output, hr)
                ssim_loss = 1 * (1 - cal_ssim(output, hr))
                perception_loss = 1 * perception_loss_fn(output, hr)
                perception_loss = torch.mean(perception_loss, dim=0).squeeze_().squeeze_().squeeze_()
                if configs.training_config['atom_loss']['enable']:
                    atom_loss = configs.training_config['atom_loss']['loss_weight'] * cal_atom_loss(atom_fn, output, hr)
                    loss = l1_loss + ssim_loss + perception_loss + atom_loss
                else:
                    loss = l1_loss + ssim_loss + perception_loss
            loss_meter.update(loss.item())
            psnr_meter.update(cal_psnr(output, hr).item())
            ssim_meter.update(cal_ssim(output, hr).item())
            lpips_meter.update(perception_loss.item())

        loop.set_description('Validation Epoch {}'.format(epoch))
        loop.set_postfix(loss=loss_meter.avg,
                         psnr=psnr_meter.avg,
                         ssim=ssim_meter.avg,
                         lpips=lpips_meter.avg)
    val_loss = loss_meter.avg
    val_psnr = psnr_meter.avg
    val_ssim = ssim_meter.avg
    val_lpips = lpips_meter.avg
    return val_loss, val_psnr, val_ssim, val_lpips


def save(train_log, val_log, model, savename, epoch, start_epoch):
    train_logs = pd.DataFrame({'epoch': train_log.epoch_list,
                               'loss': train_log.loss_list,
                               'psnr': train_log.psnr_list,
                               'ssim': train_log.ssim_list,
                               'lpips': train_log.lpips_list,
                               'lr': train_log.lr_list,
                               })
    val_logs = pd.DataFrame({'epoch': val_log.epoch_list,
                             'val_loss': val_log.loss_list,
                             'val_psnr': val_log.psnr_list,
                             'val_ssim': val_log.ssim_list,
                             'val_lpips': val_log.lpips_list,
                             })
    if not os.path.exists("./logs/{}".format(savename)):
        os.mkdir("./logs/{}".format(savename))

    train_logs.to_csv('./logs/{}/{}_train.csv'.format(savename, savename), index=False)
    val_logs.to_csv('./logs/{}/{}_val.csv'.format(savename, savename), index=False)

    if not os.path.exists("./weights/{}".format(savename)):
        os.mkdir("./weights/{}".format(savename))
    torch.save(model.state_dict(),
               "./weights/{}/{}_epoch{}.pt".format(savename, savename, start_epoch + epoch))

    plt.subplot(221)
    plt.title('Loss')
    plt.plot(train_log.epoch_list, train_log.loss_list, label='Training')
    if configs.training_config['validation']['enable']:
        plt.plot(val_log.epoch_list, val_log.loss_list, label='Validation')
    plt.legend()

    plt.subplot(222)
    plt.title('LPIPS')
    plt.plot(train_log.epoch_list, train_log.lpips_list, label='Training')
    if configs.training_config['validation']['enable']:
        plt.plot(val_log.epoch_list, val_log.lpips_list, label='Validation')
    plt.legend()

    plt.subplot(223)
    plt.title('PSNR')
    plt.plot(train_log.epoch_list, train_log.psnr_list, label='Training')
    if configs.training_config['validation']['enable']:
        plt.plot(val_log.epoch_list, val_log.psnr_list, label='Validation')
    plt.legend()

    plt.subplot(224)
    plt.title('SSIM')
    plt.plot(train_log.epoch_list, train_log.ssim_list, label='Training')
    if configs.training_config['validation']['enable']:
        plt.plot(val_log.epoch_list, val_log.ssim_list, label='Validation')
    plt.legend()

    plt.tight_layout()
    plt.savefig('./logs/{}/{}_figure.png'.format(savename, savename), dpi=300)
    plt.clf()  # 清空画布防止legend堆叠
