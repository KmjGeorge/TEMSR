import pytorch_ssim
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

device = configs.device


class TrainingLog:
    def __init__(self):
        self.loss_list = []
        self.psnr_list = []
        self.ssim_list = []
        self.val_loss_list = []
        self.val_psnr_list = []
        self.val_ssim_list = []
        self.lr_list = []

    def update(self, loss, psnr, ssim, val_loss, val_psnr, val_ssim, lr):
        self.loss_list.append(loss)
        self.psnr_list.append(psnr)
        self.ssim_list.append(ssim)
        self.val_loss_list.append(val_loss)
        self.val_psnr_list.append(val_psnr)
        self.val_ssim_list.append(val_ssim)
        self.lr_list.append(lr)


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
    log = TrainingLog()
    optim_config = training_config['optim_config']
    scheduler_config = training_config['scheduler_config']
    optimizer = torch.optim.Adam(model.parameters(), lr=optim_config['lr'],
                                 weight_decay=optim_config['weight_decay'])
    scheduler_cos = CosineAnnealingLR(optimizer,
                                      T_max=training_config['epochs'] - scheduler_config['down_epoch'],
                                      eta_min=scheduler_config['eta_min'])
    scheduler = GradualWarmupScheduler(optimizer, multiplier=scheduler_config['multiplier'],
                                       total_epoch=scheduler_config['warmup_epoch'],
                                       after_scheduler=scheduler_cos)
    scheduler.step()
    scaler = GradScaler()
    for epoch in range(training_config['epochs']):
        epoch_loss, epoch_psnr, epoch_ssim = train_epoch(model, train_dataloader, optimizer, scaler, epoch, pytorch_ssim.SSIM())
        if epoch % training_config['validate_step'] == 0:
            val_loss, val_psnr, val_ssim = validate_epoch(model, val_dataloader)
            log.update(epoch_loss, epoch_psnr, epoch_ssim, val_loss, val_psnr, val_ssim,
                       optimizer.param_groups[0]['lr'])
        else:
            log.update(epoch_loss, epoch_psnr, epoch_ssim, -1, -1, -1,
                       optimizer.param_groups[0]['lr'])
        if training_config['save']['enable']:
            if epoch % training_config['save']['step'] == 0:
                save(log, model, epoch=epoch, savename=training_config['task_name'],
                     start_epoch=training_config['start_epoch'])

        scheduler.step()


def train_epoch(model, train_dataloader, optimizer, scaler, epoch, loss_fn):
    model.train()
    loop = tqdm(train_dataloader)
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    for lr, hr, _ in loop:
        lr = (lr.float() / 255.0).to(device)
        hr = (hr.float() / 255.0).to(device)

        optimizer.zero_grad()

        with autocast():
            output = model(lr).to(device)
            l1_loss = F.smooth_l1_loss(output, hr)
            ssim_loss = 0.1 * (1-cal_ssim(output, hr))
            # perception_fn = PerceptualLoss(blocks=[4, ], weights=[1, ], device=device)
            # perception_loss = 0.001 * perception_fn(output, hr)
            loss = l1_loss + ssim_loss

            with torch.no_grad():
                loss_meter.update(loss.item())
                psnr_meter.update(cal_psnr(output, hr).item())
                ssim_meter.update(cal_ssim(output, hr).item())


        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        loop.set_description('Epoch{}'.format(epoch))
        loop.set_postfix(loss=loss_meter.avg,
                         psnr=psnr_meter.avg,
                         ssim=ssim_meter.avg,
                         lr=optimizer.param_groups[0]['lr'])
    epoch_loss = loss_meter.avg
    epoch_psnr = psnr_meter.avg
    epoch_ssim = ssim_meter.avg
    return epoch_loss, epoch_psnr, epoch_ssim


def validate_epoch(model, val_dataloader):
    model.eval()
    loop = tqdm(val_dataloader)
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    for lr, hr, _ in loop:
        lr = (lr.float() / 255.0).to(device)
        hr = (hr.float() / 255.0).to(device)

        with autocast():
            output = model(lr).to(device)
            l1_loss = 5 * F.smooth_l1_loss(output, hr)
            ssim_loss = 0.002 * (1 - cal_ssim(output, hr))
            perception_fn = PerceptualLoss(blocks=[5, ], weights=[1, ], device=device)
            perception_loss = 0.001 * perception_fn(output, hr)
            loss = l1_loss + ssim_loss + perception_loss

        loss_meter.update(loss.item())
        psnr_meter.update(cal_psnr(output, hr).item())
        ssim_meter.update(cal_ssim(output, hr).item())

        loop.set_description('Validation')
        loop.set_postfix(loss=loss_meter.avg,
                         psnr=psnr_meter.avg,
                         ssim=ssim_meter.avg)
    val_loss = loss_meter.avg
    val_psnr = psnr_meter.avg
    val_ssim = ssim_meter.avg

    return val_loss, val_psnr, val_ssim


def save(log, model, savename, epoch, start_epoch):
    logs = pd.DataFrame({'loss': log.loss_list,
                         'psnr': log.psnr_list,
                         'ssim': log.ssim_list,
                         'val_loss': log.val_loss_list,
                         'val_psnr': log.val_psnr_list,
                         'val_ssim': log.val_ssim_list,
                         'lr': log.lr_list,
                         })
    logs.to_csv('../logs/{}_logs.csv'.format(savename), index=True)
    torch.save(model.state_dict(),
               "../weights/{}_epoch{}.pt".format(savename, start_epoch + epoch))
