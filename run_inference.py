import lpips
import torch

from dataset.temimagenet import get_temimagenet_trainval
from models.SwinIR import get_swinir
from models.UHDFour import get_uhdfour
from models.DRAN import get_dran
from train.train import validate_epoch
import configs
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import cal_ssim, cal_psnr, cal_atom_loss, atom_loss_fn
from utils.functions import image_for_network, image_for_draw
import torch.nn.functional as F
if __name__ == '__main__':
    savename = 'denosing+debg+sr2x_uhdfour_4loss 3e-5'
    model = get_uhdfour().to(configs.device)
    model.load_state_dict(torch.load('./weights/{}/{}_epoch30.pt'.format(savename, savename)))
    model.eval()


    filename = '00161'
    pic = Image.open('D:/Datasets/TEM-ImageNet-v1.3-master/image_NEAREST_128/{}.png'.format(filename)).resize((128, 128),
                                                                                                  Image.NEAREST).convert(
        'L')
    gt = Image.open('D:/Datasets/TEM-ImageNet-v1.3-master/noBackgroundnoNoise/{}.png'.format(filename)).resize(
        (256, 256), Image.Resampling.LANCZOS).convert('L')
    gt_tensor = image_for_network(gt, target_c=3)
    # gt_tensor = (torch.from_numpy(np.array(gt)) / 255.0).unsqueeze_(0).to('cuda')
    pic_tensor = image_for_network(pic, target_c=3)

    # pic = np.array(pic)
    # pic_tensor = torch.from_numpy(pic).float() / 255.0
    # pic_tensor.unsqueeze_(0)
    # pic_tensor.unsqueeze_(0)
    # pic_tensor = pic_tensor.to('cuda')

    output = model(pic_tensor)
    # out_pic = pic_tensor.detach().cpu().squeeze().squeeze().numpy()
    out_pic = image_for_draw(output)
    plt.subplot(131)
    plt.imshow(pic, 'gray')
    plt.title('Original')
    plt.subplot(132)
    plt.imshow(out_pic, 'gray')
    plt.title('Output')
    plt.subplot(133)
    plt.imshow(gt, 'gray')
    plt.title('GT')
    plt.tight_layout()
    plt.show()

    lpips_func = lpips.LPIPS(net='vgg').to(configs.device)
    atom_func = atom_loss_fn()
    print('psnr =', cal_psnr(output, gt_tensor).item())
    print('ssim =', cal_ssim(output, gt_tensor).item())
    print('l1 = ', F.smooth_l1_loss(output, gt_tensor).item() * 5)
    print('lpips = ', torch.mean(lpips_func(output, gt_tensor), dim=0).squeeze_().squeeze_().squeeze_().item())
    print('atom loss = ', cal_atom_loss(atom_func, output, gt_tensor).item() * 100)

