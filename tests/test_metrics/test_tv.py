import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2
from tqdm import tqdm
import os

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def test_tv_folder(img_folder):
    tv_avg = 0
    tvloss = TVLoss()
    for filename in tqdm(os.listdir(img_folder)):
        img = cv2.imread(os.path.join(img_folder, filename), 0).astype(float) / 255.
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        tv = tvloss(img).item()
        tv_avg += tv
    tv_avg /= len(os.listdir(img_folder))
    print('tv_avg =', tv_avg)
    return tv_avg

if __name__ == '__main__':
    # img_folder = 'F:\Datasets\Sim ReSe2\\all_crops'
    img_folder = 'F:\Datasets\Sim ReSe2\simval\GT'
    test_tv_folder(img_folder)
