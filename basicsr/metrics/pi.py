from basicsr.utils.registry import METRIC_REGISTRY
import pyiqa
import torch
from PIL import Image

@METRIC_REGISTRY.register()
def calculate_pi(img, **kwargs):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pi = pyiqa.create_metric('pi', device=device)
    pil_img = Image.fromarray(img)
    pi_result = pi(pil_img).item()
    return pi_result
