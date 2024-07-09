import argparse
import cv2
import glob
import numpy as np
import os
import torch
import shutil
from basicsr.archs.NAFNet_arch import NAFNet
from basicsr.archs.vit_arch import vit_base_patch16_gray
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa=E251
         r'F:\github\TEMSR\experiments\NAFNet-1118_1_1111_p256b16_Exp2kdenoise fft0.02 enlarge10\models\net_g_17000.pth'  # exp
        # r'F:\github\TEMSR\experiments\NAFNet-1118_1_1111_p256b16_TEMImagaNET1000denoise fft0.2 enlarge10\models\net_g_25000.pth'  # sim
    )
    parser.add_argument('--input', type=str, default=r'F:\Datasets\S2',
                        help='output folder')
    parser.add_argument('--output', type=str, default='../show/NAFNet-1118_1_1111_p256b16_Exp2kdenoise fft0.02 enlarge10 S2 exp',
                        help='output folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model

    model = NAFNet(
        width=32,
        enc_blk_nums=[1, 1, 1, 8],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
        img_channel=1)

    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    os.makedirs(args.output, exist_ok=True)
    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        img_ori = cv2.imread(path, 0)
        img = (torch.from_numpy(img_ori) / 255.0).float()
        img = img.unsqueeze(0).unsqueeze(0).to(device)
        # inference
        try:
            with torch.no_grad():
                output = model(img)
        except Exception as error:
            print('Error', error, imgname)
        else:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(args.output, f'{imgname}.png'), output)
            # cv2.imwrite(os.path.join(args.output, f'{imgname}_LQ.png'), img_ori)
            # shutil.copy(os.path.join(args.input.replace('LQ', 'HQ'), f'{imgname}.png'),
            #             os.path.join(args.output, f'{imgname}_HQ.png'))


if __name__ == '__main__':
    main()
