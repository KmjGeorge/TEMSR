import torch
import yaml
import os.path as osp
from basicsr import parse_options
from basicsr.archs import rcan_internal_arch
from basicsr.archs import rcan_arch
from torchsummary import summary

def main(root_path):
    opt, args = parse_options(root_path, is_train=True)
    print(dict(opt['network_g']))
    model = rcan_internal_arch.RCAN_Internal(
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_group=6,
                 num_block=10,
                 squeeze_factor=16,
                 upscale=2,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.198, 0.198, 0.198)).cuda()
    model.load_state_dict(torch.load('../experiments/101_RCANx2_scratch_TEMImageNet_bat8_inc3feature64group6block10_lr1e-4/models/net_g_100000.pth')['params'], strict=False)
    summary(model, input_size=(3, opt['datasets']['train']['gt_size']//opt['scale'], opt['datasets']['train']['gt_size']//opt['scale']))


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    main(root_path)