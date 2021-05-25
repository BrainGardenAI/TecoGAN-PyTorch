import os
import os.path as osp

import cv2
import numpy as np
import torch

from .base_dataset import BaseDataset
from utils.data_utils import create_kernel
from utils.base_utils import retrieve_files


class ValidationDataset(BaseDataset):
    def __init__(self, data_opt, **kwargs):
        """ Folder dataset with paired data
            support both BI & BD degradation
        """
        super(ValidationDataset, self).__init__(data_opt, **kwargs)

        # get keys
        gt_keys = sorted(os.listdir(self.gt_seq_dir))
        self.keys = sorted(list(set(gt_keys)))
        if data_opt['name'].startswith('Actors'):
            for i, k in enumerate(self.keys):
                self.keys[i] = k + '/frames'

        self.kernel = create_kernel({
            'dataset':{
                'degradation': {
                    'sigma': self.sigma
                }
            },
            'device': 'cuda'
        })

        # filter keys
        if self.filter_file:
            with open(self.filter_file, 'r') as f:
                sel_keys = { line.strip() for line in f }
                self.keys = sorted(list(sel_keys & set(self.keys)))

    def __len__(self):
        return len(self.keys)
    
    def apply_BD(self, frame_sequence):
        frame_sequence = frame_sequence.float() / 255.0
        frame_sequence = frame_sequence.permute(0, 1, 4, 2, 3)
        opt = {
            'dataset': {
                'degradation': {
                    'type': 'BD',
                    'sigma': self.sigma
                }
            },
            'scale': self.scale,
            'device': 'cuda' 
        }
        item = {'gt': frame_sequence}
        result = self.data_preparation_method(opt, item, self.kernel, batch_size=10, return_gt_data=False)
        result['lr'] = result['lr'].permute(0, 1, 3, 4, 2)
        return result

    def __getitem__(self, item):
        key = self.keys[item]
        # load gt frames and generate lr frames
        gt_seq = []
        for frm_path in retrieve_files(osp.join(self.gt_seq_dir, key)):
            frm = cv2.imread(frm_path)[..., ::-1]
            gt_seq.append(frm)
            # print(len(gt_seq))
        gt_seq = np.stack(gt_seq)  # thwc|rgb|uint8

        # convert to tensor
        gt_tsr = torch.from_numpy(np.ascontiguousarray(gt_seq))  # uint8
        # lr_tsr = self.apply_BD(gt_tsr)  # float32
        # gt: thwc|rgb||uint8 | lr: thwc|rgb|float32
        return {
            'gt': gt_tsr,
            'seq_idx': key,
            'frm_idx': sorted(os.listdir(osp.join(self.gt_seq_dir, key)))
        }
