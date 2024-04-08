from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_ssim
from .pi import calculate_pi
from .snr import calculate_snr
from .tv import calculate_tv
__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe', 'calculate_snr', 'calculate_tv']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)  # 根据yml中metrics的类型，用METRIC_REGISTRY.get调用对应和函数
    return metric
