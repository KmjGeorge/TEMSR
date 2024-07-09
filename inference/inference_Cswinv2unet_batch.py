import argparse
import shutil

import cv2
import glob
import numpy as np
import os
import torch

from basicsr.archs.ConditionalSwinv2UNet_arch import ConditionalSwinv2UNet
from basicsr.archs.instructir_arch import InstructIR
from basicsr.archs.rcan_arch import RCAN
from basicsr.data.instructir_dataset import LanguageModel, LMHead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        r'F:\github\TEMSR\experiments\CSwinUNet_p256w16b16_2gpu\models\net_g_50000.pth'
    )
    parser.add_argument("--lm_path", type=str, default='../models/lm_head/model_head_512_epoch40.pth',
                        help='embedding model head path')
    parser.add_argument("--input", type=str, default="F:\Datasets\jiangzao")
    parser.add_argument('--output', type=str, default='../show/CSwinUNet_p256w16b16_2gpu jiangzao',
                        help='output folder')
    parser.add_argument("--prompt", type=str, default='The contrast is low, please enhance it.')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = ConditionalSwinv2UNet(img_size=512,
                                  patch_size=4,
                                  in_chans=1,
                                  num_classes=1,
                                  embed_dim=128,
                                  depths=[2, 2, 18, 2],
                                  depths_decoder=[2, 2, 2, 2],
                                  num_heads=[4, 8, 16, 32],
                                  window_size=16,
                                  mlp_ratio=4,
                                  qkv_bias=True,
                                  drop_rate=0,
                                  drop_path_rate=0.1,
                                  ape=False,
                                  patch_norm=True,
                                  freeze_encoder=True).cuda()
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    '''
    model2 = InstructIR(img_channel=1, width=32, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=4, dec_blk_nums=[2, 2, 2, 2],
                         txtdim=256)
    model2.load_state_dict(torch.load('F:\github\TEMSR\experiments\InstructIR_p256b8\models\\net_g_100000.pth')['params'])
    model2.eval()
    model2 = model2.to(device)
    '''

    embedding_model = LanguageModel('../models/bge-micro-v2').eval()
    lm_head = LMHead(embedding_dim=384, hidden_dim=512, num_classes=4).eval()
    lm_head.load_state_dict(torch.load(args.lm_path))
    os.makedirs(args.output, exist_ok=True)

    cls_dict = {
        'Denoise': 0,
        'Deblur': 1,
        'LLIE': 2,
        'Segmentation': 3,
    }
    inverse_dict = dict([val, key] for key, val in cls_dict.items())
    filefolder = args.input
    for filename in os.listdir(filefolder):
        prompt = args.prompt
        # read image
        img = cv2.imread(os.path.join(filefolder, filename), 0).astype(np.float32) / 255.
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0).unsqueeze(0).to(device)
        prompt_embed, cls = lm_head(embedding_model(prompt))
        prompt_embed = prompt_embed.to(device)
        pred_cls = cls.argmax(dim=1).detach().cpu().numpy()[0]
        # print(pred_cls)
        print('Class:', inverse_dict[pred_cls])
        # inference
        try:
            with torch.no_grad():
                output = model(img, prompt_embed)
                # output2 = model2(img, prompt_embed)
        except Exception as error:
            print(error)

        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = (output * 255.0).round().astype(np.uint8)

        if not os.path.exists(os.path.join(args.output, filename)):
            shutil.copy(os.path.join(args.input, filename), os.path.join(args.output, filename))
        save_path = os.path.join(args.output,
                                 filename.replace('.jpg', '_CSwinv2UNet_{}_{}.jpg'.format(prompt,inverse_dict[pred_cls])))
        cv2.imwrite(save_path, output)
        # cv2.imwrite(save_path.replace('.jpg', '_old.jpg'), output2)
        print('Saving output image to {}'.format(save_path))


if __name__ == '__main__':
    main()
