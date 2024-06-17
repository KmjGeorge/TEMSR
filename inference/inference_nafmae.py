import argparse
import cv2
import glob
import numpy as np
import os
import torch
import math
from basicsr.archs.instructir_arch import InstructIR
from basicsr.archs.vit_arch import vit_base_patch16_gray, VisionTransformer
from functools import partial
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa=E251
        r'F:\github\TEMSR\experiments\NAFNet-1118_1_1111-withMAE-STEM_p224_b32\models\net_g_20000.pth'

    )
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--input', type=str, default=r'D:\Datasets\STEMEXP\test',
                        help='input test image folder')
    parser.add_argument('--output', type=str, default=r'D:\Datasets\STEMEXP\test_out', help='output folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model

    model = InstructIR(
        width=32,
        enc_blk_nums=[1, 1, 1, 8],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
        img_channel=1,
        txtdim=768)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    model_mae = VisionTransformer(in_chans=1, img_size=args.img_size,
                                  patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)
    model_mae.load_state_dict(torch.load(r'F:/Project/mae-main/output_dir/mae_base(DF)/checkpoint-99.pth')['model'],
                              strict=False)

    os.makedirs(args.output, exist_ok=True)
    # opt = {
    #     'tile': {'tile_size': 224,
    #              'tile_pad': 0}
    # }
    opt = None
    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        img = cv2.imread(path, 0).astype(np.float32)
        img = (torch.from_numpy(img) / 255.0).float()
        img = img.unsqueeze(0).unsqueeze(0).to(device)
        # inference
        try:
            if opt:
                batch, channel, height, width = img.shape
                output_height = height
                output_width = width
                output_shape = (batch, channel, output_height, output_width)

                # start with black image
                output = img.new_zeros(output_shape)
                tiles_x = math.ceil(width / opt['tile']['tile_size'])
                tiles_y = math.ceil(height / opt['tile']['tile_size'])

                # loop over all tiles
                for y in range(tiles_y):
                    for x in range(tiles_x):
                        # extract tile from input image
                        ofs_x = x * opt['tile']['tile_size']
                        ofs_y = y * opt['tile']['tile_size']
                        # input tile area on total image
                        input_start_x = ofs_x
                        input_end_x = min(ofs_x + opt['tile']['tile_size'], width)
                        input_start_y = ofs_y
                        input_end_y = min(ofs_y + opt['tile']['tile_size'], height)

                        # input tile area on total image with padding
                        input_start_x_pad = max(input_start_x - opt['tile']['tile_pad'], 0)
                        input_end_x_pad = min(input_end_x + opt['tile']['tile_pad'], width)
                        input_start_y_pad = max(input_start_y - opt['tile']['tile_pad'], 0)
                        input_end_y_pad = min(input_end_y + opt['tile']['tile_pad'], height)

                        # input tile dimensions
                        input_tile_width = input_end_x - input_start_x
                        input_tile_height = input_end_y - input_start_y
                        tile_idx = y * tiles_x + x + 1
                        input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                        # inference tile
                        try:
                            with torch.no_grad():
                                tile_norm = (input_tile - 0.2187) / 0.1075
                                print(tile_norm.max(), tile_norm.min())
                                feature = model_mae(tile_norm)
                                output_tile = model(input_tile, feature)
                        except RuntimeError as error:
                            print('Error', error)
                        print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                        # output tile area on total image
                        output_start_x = input_start_x
                        output_end_x = input_end_x
                        output_start_y = input_start_y
                        output_end_y = input_end_y

                        # output tile area without padding
                        output_start_x_tile = (input_start_x - input_start_x_pad)
                        output_end_x_tile = output_start_x_tile + input_tile_width
                        output_start_y_tile = (input_start_y - input_start_y_pad)
                        output_end_y_tile = output_start_y_tile + input_tile_height

                        # put tile into output image
                        output[:, :, output_start_y:output_end_y,
                        output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                       output_start_x_tile:output_end_x_tile]
            else:
                with torch.no_grad():
                    img_norm = (img - 0.2187) / 0.1075
                    feature = model_mae(img_norm)
                    output = model(img, feature)

        except Exception as error:
            print('Error', error, imgname)
        else:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(args.output, f'{imgname}_NAFNet_mae.png'), output)


if __name__ == '__main__':
    main()
