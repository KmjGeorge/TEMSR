import argparse
import cv2
import glob
import numpy as np
import os
import torch

from basicsr.archs.instructir_arch import InstructIR
from basicsr.archs.rcan_arch import RCAN
from basicsr.data.instructir_dataset import LanguageModel, LMHead


def main():
    os.environ["http_proxy"] = 'http://127.0.0.1:7890'
    os.environ["https_proxy"] = 'http://127.0.0.1:7890'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        'F:/github/TEMSR/experiments/InstructIR_p256b8/models//net_g_100000.pth'
    )
    parser.add_argument("--lm_path", type=str, default='F:/Project/InstructIR-main/models/ft/model_head_60.pth', help='embedding model head path')
    parser.add_argument('--output', type=str, default='F:/github/TEMSR/results/InstructTEMSR',
                        help='output folder')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = InstructIR(img_channel=1, width=32, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=4, dec_blk_nums=[2, 2, 2, 2],
                       txtdim=256)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    embedding_model = LanguageModel('TaylorAI/bge-micro-v2').eval()
    lm_head = LMHead(embedding_dim=384, hidden_dim=256, num_classes=5).eval()
    lm_head.load_state_dict(torch.load(args.lm_path))
    os.makedirs(args.output, exist_ok=True)

    cls_dict = {
        'Denoise': 0,
        'Deblur': 1,
        'Depollute': 2,
        'Low-Light': 3,
        'Enhancement': 4,
    }
    inverse_dict = dict([val, key] for key, val in cls_dict.items())


    filepath = input('File Path:')
    while True:
        if filepath == 'Exit':
            break
        if os.path.exists(filepath):
            prompt = input('Prompt:')
            # read image
            img = cv2.imread(filepath, 0).astype(np.float32) / 255.
            img = torch.from_numpy(img).float()
            img = img.unsqueeze(0).unsqueeze(0).to(device)
            prompt_embed, cls = lm_head(embedding_model(prompt))
            prompt_embed = prompt_embed.to(device)
            pred_cls = cls.argmax(dim=1).detach().cpu().numpy()[0]
            print(pred_cls)
            print('Class:', inverse_dict[pred_cls])
            # inference
            try:
                with torch.no_grad():
                    output = model(img, prompt_embed)
            except Exception as error:
                print('Error', error, filepath)
            else:
                # save image
                output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                # output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
                output = (output * 255.0).round().astype(np.uint8)
                cv2.imwrite(os.path.join(args.output, os.path.split(filepath)[1].replace('.png', '_InstructIR.png')), output)
                print('Saving output image to {}'.format(os.path.join(args.output, os.path.split(filepath)[1].replace('.png', '_InstructIR.png'))))
        else:
            print('Path Error!')
        filepath = input('File Path:')

if __name__ == '__main__':
    main()
