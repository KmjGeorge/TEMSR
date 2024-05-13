import random
import numpy as np
import pandas as pd
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizer, AutoModel, AutoTokenizer
import os


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class LanguageModel(nn.Module):
    def __init__(self, model='../models/bge-micro-v2'):
        super(LanguageModel, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
        self.model_name = model
        # Remove the CLIP vision tower
        if "clip" in self.model_name:
            self.model.vision_model = None
        # Freeze the pre-trained parameters (very important)
        for param in self.model.parameters():
            param.requires_grad = False

        # Make sure to set evaluation mode (also important)
        self.model.eval()

    def forward(self, text_batch):
        inputs = self.tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():  # Ensure no gradients are computed for this forward pass

            if "clip" in self.model_name:
                sentence_embedding = self.model.get_text_features(**inputs)
                return sentence_embedding

            outputs = self.model(**inputs)
        sentence_embeddings = mean_pooling(outputs, inputs['attention_mask'])
        # Normalize embeddings
        sentence_embedding = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embedding


class LMHead(nn.Module):
    def __init__(self, embedding_dim=384, hidden_dim=256, num_classes=4):
        super(LMHead, self).__init__()

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        # self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embd = self.fc1(x)
        embd = F.normalize(embd, p=2, dim=1)
        deg_pred = self.fc2(embd)
        return embd, deg_pred


@DATASET_REGISTRY.register()
class InstructIRDataset(data.Dataset):

    def __init__(self, opt):
        super(InstructIRDataset, self).__init__()
        self.opt = opt
        os.environ["http_proxy"] = self.opt['http_proxy']
        os.environ["https_proxy"] = self.opt['http_proxy']
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        # meta_info : <lq_path> <gt_path> <instruct> <class>
        self.meta_info = pd.read_csv(opt['meta_info'])
        self.gt_paths, self.lq_paths, self.classes = self.meta_info['gt_path'], self.meta_info['lq_path'], self.meta_info['class']

        # # instruct file: <instruct> <class>
        # if opt.get('instruct_file_path'):
        #     self.instruct_data = pd.read_csv(opt['instruct_file_path'])

        self.embedding_model = LanguageModel(self.opt['lm_path']).eval()
        self.lm_head = LMHead(embedding_dim=384, hidden_dim=256, num_classes=5).eval()
        self.lm_head.load_state_dict(torch.load(opt['lm_head_weight_path']))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.gt_paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.lq_paths[index]

        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        lq_cls = self.classes[index]

        lq_instruct = self.meta_info['instruct'][index]

        ### here
        if 'TEM ' in lq_instruct:
            lq_instruct = lq_instruct.replace('TEM ', 'STEM ')

        if 'MoS2' in lq_instruct:
            lq_instruct = lq_instruct.replace('MoS2', 'ReSe2')

        if 'ReS2' in lq_instruct:
            lq_instruct = lq_instruct.replace('ReS2', 'ReSe2')
        ###


        with torch.no_grad():
            instruct_embd, _ = self.lm_head(self.embedding_model(lq_instruct))
        instruct_embd.squeeze_()   # (256,)
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]
        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        if 'color' in self.opt and self.opt['color'] == 'gray':
            img_lq = img_lq[0].unsqueeze(0)
            img_gt = img_gt[0].unsqueeze(0)

        return {'instruct_embd': instruct_embd, 'instruct_cls': lq_cls, 'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path,
                'gt_path': gt_path}

    def __len__(self):
        return len(self.lq_paths)
