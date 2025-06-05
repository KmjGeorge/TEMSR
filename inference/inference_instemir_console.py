import argparse
import shutil

import cv2
import numpy as np
import os
import torch

from basicsr.archs.DegAE_InstructIR_arch import DegAE_InstructIR
from basicsr.data.instructir_dataset import LanguageModel, LMHead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        r'F:\github\TEMSR\experiments\DegAE_FT\DegAE_InstructIR_TEM_t256_p128b4_full\models\net_g_480000.pth'
    )
    parser.add_argument("--lm_path", type=str, default=r'F:\github\TEMSR\models\lm_head\model_head_256_dn_db_dd_ll_epoch90.pth',
                        help='embedding model head path')
    parser.add_argument('--output', type=str, default='../show/InsTEMIR_full_48w_tSNE',
                        help='output folder')
    # parser.add_argument('--input', type=str, default=r'D:\Datasets\DegAEFT\DeBG(with Pollution)\train\img781 (2)_s002_STEM_atom.png',
    #                     help='output folder')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = DegAE_InstructIR(inp_channels=1, head_channels=64, dim=48, num_blocks=[4, 6, 6, 8],
                             heads=[1,2,4,8], num_refinement_blocks=4, ffn_expansion_factor=2.66, bias=False,
                             LayerNorm_type='WithBias', dual_pixel_task=False, global_residual=False,
                             text_dim=256, decoder_blk_nums=[2, 2, 2, 2])
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)


    embedding_model = LanguageModel('../models/bge-micro-v2').eval()
    lm_head = LMHead(embedding_dim=384, hidden_dim=256, num_classes=4).eval()
    lm_head.load_state_dict(torch.load(args.lm_path))
    os.makedirs(args.output, exist_ok=True)

    cls_dict = {
        'Denoise': 0,
        'De-Background': 1,
        'De-Distortion': 2,
        'LCIE': 3,
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
                if '.jpg' in filename:
                    filename = filename.replace('.jpg', '.png')
                save_path = os.path.join(args.output, filename.replace('.png','_InsTEMIR_{}.png'.format(inverse_dict[pred_cls])))
                cv2.imwrite(save_path,  output)
                # cv2.imwrite(save_path.replace('.png', '_old.png'), output2)
                print('Saving output image to {}'.format(save_path))
        else:
            print('Path Error!')
        filename = input('File name:')


if __name__ == '__main__':
    main()
