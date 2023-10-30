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
    savename = 'denosing+debg+sr_dran_3loss 3e-5'

    model = get_dran().to(configs.device)

    model.load_state_dict(torch.load('./weights/{}/{}_epoch30.pt'.format(savename, savename)))
    model.eval()

    # no ground truth
    filename = '0_0_14'

    pic = Image.open('D:\Datasets\TEMSlide_new\slide_256(128)_3class_orig\\train\images\{}.jpg'.format(filename)).resize((128, 128),Image.NEAREST).convert('L')
    pic_tensor = image_for_network(pic, target_c=3)
    output = model(pic_tensor)
    out_pic = image_for_draw(output)
    plt.subplot(121)
    plt.imshow(pic, 'gray')
    plt.title('Original')
    plt.subplot(122)
    plt.imshow(out_pic, 'gray')
    plt.title('Output')
    plt.tight_layout()
    plt.savefig('{}.png'.format(filename), dpi=300)
    plt.show()
    out_pic_img = Image.fromarray(np.uint8(out_pic * 255.0))
    out_pic_img.save('{}_output.png'.format(filename))
    assert False


    # with ground truth
    filename = '04830'
    # pic = Image.open('D:/Datasets/TEM-ImageNet-v1.3-master/image_NEAREST_128/{}.png'.format(filename)).resize((128, 128),
    #                                                                                               Image.NEAREST).convert(
    #     'L')
    pic = Image.open('D:/Datasets/TEM-ImageNet-v1.3-master/image/{}.png'.format(filename)).convert('L')
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
    plt.savefig('{}.png'.format(filename), dpi=300)
    plt.show()

    outpic_img = Image.fromarray(np.uint8(out_pic * 255))

    outpic_img.save('{}_output.png'.format(filename))

    lpips_func = lpips.LPIPS(net='vgg').to(configs.device)
    atom_func = atom_loss_fn()
    print('psnr =', cal_psnr(output, gt_tensor).item())
    print('ssim =', cal_ssim(output, gt_tensor).item())
    print('l1 = ', F.smooth_l1_loss(output, gt_tensor).item() * 5)
    print('lpips = ', torch.mean(lpips_func(output, gt_tensor), dim=0).squeeze_().squeeze_().squeeze_().item())
    # print('atom loss = ', cal_atom_loss(atom_func, output, gt_tensor).item() * 100)

