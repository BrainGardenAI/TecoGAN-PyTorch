import os
import os.path as osp

import cv2
import numpy as np
import torch

from .base_dataset import BaseDataset
from utils.base_utils import retrieve_files


class SimpleDataset(BaseDataset):
    def __init__(self, data_opt, **kwargs):
        """ Folder dataset with paired data
            support both BI & BD degradation
        """
        super(SimpleDataset, self).__init__(data_opt, **kwargs)

        # get keys
        if data_opt['name'].startswith('Actors'):
            self.gt_seq_dir = self.gt_seq_dir + 'frames/'
        gt_keys = sorted(os.listdir(self.gt_seq_dir))
        self.keys = sorted(list(set(gt_keys)))
        self.keys = [k for k in self.keys if k.endswith('jpg') or k.endswith('png')]

        # filter keys
        if self.filter_file:
            with open(self.filter_file, 'r') as f:
                sel_keys = { line.strip() for line in f }
                self.keys = sorted(list(sel_keys & set(self.keys)))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        key = self.keys[item]
        frm = cv2.imread(osp.join(self.gt_seq_dir, key))[..., ::-1].transpose(2, 0, 1)
        frm = torch.FloatTensor(np.ascontiguousarray(frm)).unsqueeze(0) / 127.5 - 1
        return {
            'gt': frm,
            'frame_key': key,
            'frame_path': self.gt_seq_dir
        }
