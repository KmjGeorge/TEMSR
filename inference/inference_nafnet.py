import argparse
import cv2
import glob
import numpy as np
import os
import torch

from basicsr.archs.NAFNet_arch import NAFNet
from basicsr.archs.vit_arch import vit_base_patch16_gray
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa=E251
        r'F:\github\TEMSR\experiments\NAFNet-1118_2_1111_STEM_p224_b32\models\net_g_20000.pth'
    )
    parser.add_argument('--input', type=str, default=r'D:\Datasets\STEMEXP\test',
                        help='input test image folder')
    parser.add_argument('--output', type=str, default=r'D:\Datasets\STEMEXP\test_out', help='output folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model

    model = NAFNet(
        width=32,
        enc_blk_nums=[1, 1, 1, 8],
        middle_blk_num=2,
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
        img = cv2.imread(path, 0).astype(np.float32) / 255.0
        img = torch.from_numpy(img).float()
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
            cv2.imwrite(os.path.join(args.output, f'{imgname}_NAFNet.png'), output)


if __name__ == '__main__':
    main()
