import argparse
import cv2
import glob
import numpy as np
import os
import torch

from basicsr.archs.rcan_arch import RCAN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        'D:/github/TEMSR_BasicSR/experiments/pretrained_models/RCAN/101_RCANx2_scratch_TEMImageNet_bat8_inc3feature64group6block10_lr1e-4_300000.pth'
    )
    parser.add_argument('--input', type=str, default='D:\Datasets\TEM-ImageNet-v1.3-master\image\\val',
                        help='input test image folder')
    parser.add_argument('--output', type=str, default='D:/github/TEMSR_BasicSR/results/RCAN/TEMImageNet/3w', help='output folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = RCAN(num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_group=6,
                 num_block=10,
                 squeeze_factor=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.1980, 0.1980, 0.1980))
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    os.makedirs(args.output, exist_ok=True)
    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)
        # inference
        try:
            with torch.no_grad():
                output = model(img)
        except Exception as error:
            print('Error', error, imgname)
        else:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(args.output, f'{imgname}_RCAN.png'), output)


if __name__ == '__main__':
    main()
