import sys
import os
import numpy as np

import torch
import matplotlib.pyplot as plt
from PIL import Image

from basicsr.archs.swinmaeir_arch import SwinMAEIR


# define the utils
def show_image(image, title=''):
    # image is [H, W, 3]
    # assert image.shape[2] == 3
    image = torch.clip(image * 255, 0, 255).int().squeeze()
    image = np.asarray(Image.fromarray(np.uint8(image)).resize((128, 128)))
    plt.imshow(image, 'gray')
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def run_one_image(x, model):
    x = torch.tensor(x)
    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float().cuda())

    # y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, 1)  # (N, H*W, p*p*c)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x)

    plt.subplot(2,3,1)
    show_image(y[0], 'pred')

    # masked image
    im_masked = x * (1 - mask)
    y = y * mask

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [12, 6]

    plt.subplot(2, 3, 2)
    show_image(x[0], "original")

    plt.subplot(2, 3, 3)
    show_image(im_masked[0], "masked")

    plt.subplot(2, 3, 4)

    show_image(y[0], "reconstruction")

    plt.subplot(2, 3, 6)
    show_image(im_paste[0], "reconstruction + visible")

    plt.show()


if __name__ == '__main__':
    # 读取图像
    img_root = r'D:\Datasets\STEMEXP\warwick'
    img_name = r'img548 (2)_s001.png'
    img_size = (56, 56)
    img = Image.open(os.path.join(img_root, img_name)).convert('L')
    img = img.crop((10, 10, 10+56, 10+56))
    img = np.asarray(img) / 255.
    img = img[..., np.newaxis]
    print(img.shape)
    # 读取模型
    model = SwinMAEIR(window_masking_r=4,
                      upscale=1,
                      img_size=(56, 56),
                      window_size=8,
                      img_range=1.,
                      depths=[6, 6, 6, 6, 6, 6],
                      embed_dim=180,
                      num_heads=[6, 6, 6, 6, 6, 6],
                      mlp_ratio=2,
                      upsampler='',
                      in_chans=1).cuda()
    model.load_state_dict(torch.load('F:/Project/mae-main/output_dir/checkpoint-20.pth')['model'])
    print('Model loaded.')

    # make random mask reproducible (comment out to make it change)
    # torch.manual_seed(2)
    print('MAE with pixel reconstruction:')
    run_one_image(img, model)
