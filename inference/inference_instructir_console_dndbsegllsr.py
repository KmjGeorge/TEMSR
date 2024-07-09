import argparse
import shutil

import cv2
import glob
import numpy as np
import os
import torch

from basicsr.archs.instructir_arch import InstructIR
from basicsr.archs.rcan_arch import RCAN
from basicsr.data.instructir_dataset import LanguageModel, LMHead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        r'F:\github\TEMSR\experiments\InstructIR_32_2248_12_2222_p256b8_tdim256_2gpu_5task fft0.05\models\net_g_175000.pth'
    )
    parser.add_argument("--lm_path", type=str, default='../models/lm_head/model_head_256_dndbseglliesr_epoch40.pth',
                        help='embedding model head path')
    parser.add_argument('--output', type=str, default='../show/InstructIR_new SR test 17w5/console',
                        help='output folder')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = InstructIR(img_channel=1, width=32, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2],
                       txtdim=256)
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
    lm_head = LMHead(embedding_dim=384, hidden_dim=256, num_classes=5).eval()
    lm_head.load_state_dict(torch.load(args.lm_path))
    os.makedirs(args.output, exist_ok=True)

    cls_dict = {
        'Denoise': 0,
        'Deblur': 1,
        'Segmentation': 2,
        'LLIE': 3,
        'SR': 4
    }
    inverse_dict = dict([val, key] for key, val in cls_dict.items())
    filefolder = input('File Folder:')
    filename = input('File name:')
    while True:
        if filename == 'Exit':
            break
        if os.path.exists(os.path.join(filefolder, filename)):
            prompt = input('Prompt:')
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
            else:
                # save image
                output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output = (output * 255.0).round().astype(np.uint8)

                # output2 = output2.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                # output2 = (output2 * 255.0).round().astype(np.uint8)

                if not os.path.exists(os.path.join(args.output, filename)):
                    shutil.copy(os.path.join(filefolder, filename), os.path.join(args.output, filename))
                save_path = os.path.join(args.output, filename.replace('.png','_InstructIR_{}.png'.format(inverse_dict[pred_cls])))
                cv2.imwrite(save_path,  output)
                # cv2.imwrite(save_path.replace('.png', '_old.png'), output2)
                print('Saving output image to {}'.format(save_path))
        else:
            print('Path Error!')
        filename = input('File name:')


if __name__ == '__main__':
    main()
