import argparse
import shutil

import cv2
import glob
import numpy as np
import os
import torch
from basicsr.archs.ConditionalSwinv2UNet_arch import ConditionalSwinv2UNet
from basicsr.archs.Swinv2UNet_arch import Swinv2UNet
from basicsr.archs.instructir_arch import InstructIR
from basicsr.archs.rcan_arch import RCAN
from basicsr.data.instructir_dataset import LanguageModel, LMHead
from torchvision import transforms


def load_pretrained_swinv2(model, load_path, params_key):
    checkpoint = torch.load(load_path)
    state_dict = checkpoint[params_key]

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            raise f"Error in loading {k}, passing......"
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match

                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
                print('relative_position_bias_table has been interpolated from {} to {}'.format(L1, L2))
    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            raise f"Error in loading {k}, passing......"
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized
                print('absolute_pos_embed bias has been interpolated from {} to {}'.format(L1, L2))

    # state_dict = OrderedDict(state_dict)
    msg = model.load_state_dict(state_dict, strict=False)
    print('Loaded pretrained Swinv2UNet')
    del checkpoint
    torch.cuda.empty_cache()


def inference_tile(model, img_tensor, tilesize, stride, device):
    # 原始图像尺寸
    height, width = img_tensor.shape[2], img_tensor.shape[3]

    # 计算需要的块的数量
    num_blocks_height = (height + stride - 1) // stride
    num_blocks_width = (width + stride - 1) // stride

    # 初始化累积图像，用于累加所有块的输出
    acc_image = torch.zeros(1, 1, height, width).cpu()

    # 初始化计数图像，用于记录每个像素被覆盖的次数
    count_image = torch.zeros(1, 1, height, width).cpu()

    # 遍历每个块
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            print('Processing Tile : {}, {}.  Total : {}, {}'.format(i + 1, j + 1, num_blocks_height, num_blocks_width))
            # 计算块的坐标
            x1, y1 = i * stride, j * stride
            x2, y2 = min(x1 + tilesize, height), min(y1 + tilesize, width)
            # 裁剪块
            block = img_tensor[:, :, x1:x2, y1:y2]

            # 填充块以适应模型输入大小
            padded_block = torch.zeros_like(img_tensor[:, :, :tilesize, :tilesize])
            padded_block[:, :, :x2 - x1, :y2 - y1] = block
            # 推理块
            block_output = model(padded_block).detach().cpu()

            # 累加输出和计数
            acc_image[:, :, x1:x2, y1:y2] += block_output[:, :, :x2 - x1, :y2 - y1]
            count_image[:, :, x1:x2, y1:y2] += 1

    # 计算平均值
    final_image = acc_image / torch.clamp(count_image, min=1)
    return final_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        r'F:\github\TEMSR\experiments\SwinUNet_ft_p256w16b16_TEM2kExp2kdenoise fft0.1 norm freeze\models\net_g_22000.pth'
        # r'F:\github\TEMSR\experiments\SwinUNet_ft_p256w16b16_TEM5kdenoise fft0.1 norm freeze\models\net_g_18000.pth'

    )
    parser.add_argument('--input', type=str,
                        # default=r'D:\Datasets\PairsEXP\LQ256\val',
                        default=r'F:\Datasets\4',
                        help='output folder')
    parser.add_argument('--output', type=str,
                        default='../show/SwinUNet_ft_p256w16b16_TEM2kExp2kdenoise fft0.1 norm freeze 2w2 4',
                        help='output folder')
    parser.add_argument("--patch_size", type=int,
                        default=512)
    parser.add_argument("--post_fix",type=str,default='Swinv2UNet')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = Swinv2UNet(img_size=args.patch_size,
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
    load_pretrained_swinv2(model, args.model_path, 'params')
    # model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    os.makedirs(args.output, exist_ok=True)

    input_folder = args.input
    # mean = 0.1682      # TEMImageNet
    # std = 0.1582

    mean = 0.2316  # TEM2k+Exp2k
    std = 0.1530

    # mean = 0.2944       # Exp2k
    # std = 0.1450

    # mean = 0
    # std = 1

    # for filename in os.listdir(input_folder):
    #     img = cv2.imread(os.path.join(input_folder, filename), 0).astype(np.float32) / 255.
    #     mean += img.mean()
    #     var += img.var()
    # mean = mean / len(os.listdir(input_folder))
    # var = var / len(os.listdir(input_folder))
    # std = np.sqrt(var)
    # print('mean =', mean, 'std =', std)
    total_length = len(os.listdir(input_folder))
    for idx, filename in enumerate(os.listdir(input_folder)):
        img_ori = cv2.imread(os.path.join(input_folder, filename), 0)
        h, w = img_ori.shape
        if h < args.patch_size:
            pad_1 = (0, args.patch_size - h)
        else:
            pad_1 = (0, 0)
        if w < args.patch_size:
            pad_2 = (0, args.patch_size - w)
        else:
            pad_2 = (0, 0)
        img = np.pad(img_ori, (pad_1, pad_2),  mode='constant', constant_values=0)
        img = img.astype(float) / 255.0
        img = (img - mean) / std
        img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
        output = inference_tile(model, img, tilesize=args.patch_size, stride=args.patch_size, device=device)
        # output = model(img)
        output = output.squeeze().detach().cpu().numpy()
        output = output[:h, :w]
        # output = (np.clip(output * std + mean, 0, 1) * 255).round().astype(np.uint8)

        # normalize to 0~1
        output_max, output_min = output.max(), output.min()
        output = (output - output_min) / (output_max - output_min)
        output = (np.clip(output, 0, 1) * 255).round().astype(np.uint8)
        if '.png' in filename:
            save_path = os.path.join(args.output,
                                     filename.replace('.png', '_InstructIR_{}.png'.format(inverse_dict[pred_cls])))
        else:
            save_path = os.path.join(args.output,
                                     filename.replace('.jpg', '_InstructIR_{}.png'.format(inverse_dict[pred_cls])))
        cv2.imwrite(save_path, output)

        # cv2.imwrite(os.path.join(args.output, filename.replace('.jpg', '_LQ.jpg')), img_ori)
        # shutil.copy(os.path.join(args.input.replace('LQ', 'HQ'), filename),
        #             os.path.join(args.output, filename.replace('.jpg', '_HQ.jpg')))

        # cv2.imwrite(save_path.replace('.jpg', '_old.jpg'), output2)
        # print('{} / {}: Saving output image to {}'.format(idx, total_length, save_path))


if __name__ == '__main__':
    main()
