import torch
from collections import OrderedDict
from os import path as osp

from basicsr.models.base_model import BaseModel
from matplotlib import pyplot as plt
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel
from basicsr.utils.sde_utils import IRSDE


@MODEL_REGISTRY.register()
class DiffusionModel(SRModel):
    def __init__(self, opt):  # 初始化，定义网络，读取权重
        super(DiffusionModel, self).__init__(opt)

        network_l = self.opt.get('network_l')
        if network_l:
            self.net_l = build_network(opt['network_l'])  # latent encoder-decoder
            self.net_l = self.model_to_device(self.net_g)
            l_load_path = self.opt['path'].get('pretrain_network_l', None)
            self.net_l.load_state_dict(l_load_path)
            # self.print_network(self.net_l)
        else:
            self.net_l = None
        self.sde = IRSDE(max_sigma=opt['sde']['max_sigma'],
                         T=opt['sde']['T'],
                         schedule=opt['sde']['schedule'],
                         eps=opt['sde']['eps'],
                         device=self.device)  # 实例化SDE
        self.sde.set_model(self.net_g)
        self.perform_ode = self.opt['sde'].get('perform_ode', False)

    def feed_data(self, data):  # 喂数据，是与dataloder(dataset)的接口
        lq = data['lq'].to(self.device)
        # 对lq进行双三次到目标维度
        self.lq = torch.nn.functional.interpolate(lq, scale_factor=self.opt['scale'], mode='bicubic').to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        if self.net_l:
            lq, _ = self.net_l.encode(self.lq)
            gt, _ = self.net_l.encode(self.gt)
        else:
            lq = self.lq
            gt = self.gt
        self.timesteps, self.state = self.sde.generate_random_states(x0=gt, mu=lq)
        self.state = self.state.to(self.device)  # noisy_state
        self.condition = self.lq.to(self.device)  # LQ
        self.state_0 = self.gt.to(self.device)  # GT
        self.sde.set_mu(self.condition)

        self.optimizer_g.zero_grad()
        timesteps = self.timesteps.to(self.device)

        # Get noise and score
        noise = self.sde.noise_fn(self.state, timesteps.squeeze())
        score = self.sde.get_score_from_noise(noise, timesteps)

        # Learning the maximum likelihood objective for state x_{t-1}
        xt_1_expection = self.sde.reverse_sde_step_mean(self.state, score, timesteps)
        xt_1_optimum = self.sde.reverse_optimum_step(self.state, self.state_0, timesteps)
        l_total = 0
        loss_dict = OrderedDict()
        if self.cri_pix:
            l_pix = self.cri_pix(xt_1_expection, xt_1_optimum)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
            # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(xt_1_expection, xt_1_optimum)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        l_total.backward()

        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:  # 使用EMA
            self.model_ema(decay=self.ema_decay)

    def test(self, save_states=False):
        self.state = self.sde.noise_state(self.lq)  # noisy_state
        self.condition = self.lq.to(self.device)  # LQ
        self.state_0 = self.gt.to(self.device)  # GT
        self.sde.set_mu(self.condition)
        perform_ode = self.perform_ode
        self.net_g.eval()
        with torch.no_grad():
            # self.output = sde.forward(self.state_0)
            if not perform_ode:
                # for SDE
                self.output = self.sde.reverse_sde(self.state, save_states=save_states)
            else:
                # if perform Denoising ODE
                self.output = self.sde.reverse_ode(self.state, save_states=save_states)
        self.net_g.train()
