import torch
import numpy as np
import random

import os
from os import path as osp

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Union

from .transforms.torch_transforms import transform_image



@dataclass
class Frame:
    name: str
    seq_name: str
    next_frame_idx: int
    prev_frame_idx: int


@dataclass
class Sequence:
    name: str
    first_frame: int
    len: int


class MultiModalDataset(Dataset):
    def __init__(self, data_path: str, modalities: Dict, tempo_extent: int, crop_size: int):
        super().__init__()

        self.data_path = data_path 
        self.modalities = modalities
        self.tempo_extent = tempo_extent
        self.crop_size = crop_size
        self.frame_list, self.seq_list = self._build_frame_and_seq_list(data_path, modalities)
        
        # getting width and height of the samples
        frm = self.frame_list[0]
        frm_path = osp.join(data_path, frm.seq_name, modalities["ground_truth"]["name"]) + "/" + frm.name
        img = Image.open(frm_path)
        self._w, self._h = img.size 

    def __len__(self):
        return len(self.frame_list)
    
    def __getitem__(self, idx: int):
        gt_imgs = []
        input_imgs = [[] for _ in range(self.tempo_extent)]
        curr_idx = idx
        forward = True
        bbox = self.get_random_crop(self.crop_size, self._w, self._h)
        for i in range(self.tempo_extent):
            frame = self.frame_list[curr_idx]

            gt_path = osp.join(self.data_path, frame.seq_name, self.modalities["ground_truth"]["name"])
            gt_path += "/{}.{}".format(frame.name, self.modalities["ground_truth"]["ext"])

            img = Image.open(gt_path).crop(bbox)
            img = transform_image(img, self.modalities["ground_truth"]["type"])
            gt_imgs.append(img.unsqueeze(0))

            for mod_idx in range(len(self.modalities.keys()) - 1):
                key = "input_" + str(mod_idx + 1)
                input_path = osp.join(self.data_path, frame.seq_name, self.modalities[key]["name"])
                input_path += "/{}.{}".format(frame.name, self.modalities[key]["ext"])
                img = Image.open(input_path).crop(bbox)
                img = transform_image(img, self.modalities[key]["type"])
                input_imgs[i].append(img.unsqueeze(0))

            if frame.next_frame_idx and forward:
                curr_idx = frame.next_frame_idx
            else:
                forward = False
                curr_idx = frame.prev_frame_idx

        gt_imgs = torch.cat(gt_imgs)
        for i, input_t in enumerate(input_imgs):
            input_imgs[i] = torch.cat(input_t, dim=1)
        input_imgs = torch.cat(input_imgs)
        return {'gt': gt_imgs, 'lr': input_imgs}

    def _build_frame_and_seq_list(self, data_path: str, modalities: Dict) -> Tuple[List[Frame], List[Sequence]]:
        frame_list = []
        seq_list = []
        gt_name = modalities["ground_truth"]["name"]
        seq_names = [dir for dir in os.listdir(osp.join(data_path)) if not dir.startswith(".")]
        for seq_name in seq_names:
            frame_names = list(sorted(os.listdir(osp.join(data_path, seq_name, gt_name))))
            prev_frame = None
            
            seq_list.append(Sequence(seq_name, len(frame_list), len(frame_names)))
            for frame_name in frame_names:
                frame_name = frame_name.split('.')[0]
                curr_frame = Frame(frame_name, seq_name, None, None)
                frame_list.append(curr_frame)
                if prev_frame:
                    prev_frame.next_frame_idx = len(frame_list) - 1
                    curr_frame.prev_frame_idx = len(frame_list) - 2
                prev_frame = curr_frame    

        return frame_list, seq_list
    
    @staticmethod
    def get_random_crop(crop_size, image_w, image_h):
        left = random.randint(0, image_w - crop_size)
        upper = random.randint(0, image_h - crop_size)
        return left, upper, left + crop_size, upper + crop_size


class MultiModalValidationDataset(MultiModalDataset):
    def __init__(self, data_path: str, modalities: Dict):
        super().__init__(data_path, modalities, None, None)
    
    def __len__(self):
        return len(self.seq_list)
    
    def __getitem__(self, idx):
        sequence = self.seq_list[idx]
        gt_imgs = []
        input_imgs = [[] for _ in range(sequence.len)]
        curr_idx = sequence.first_frame
        frame_idx = []
        for i in range(sequence.len):
            frame = self.frame_list[curr_idx]
            frame_idx.append(frame.name)

            gt_path = osp.join(self.data_path, frame.seq_name, self.modalities["ground_truth"]["name"])
            gt_path += "/{}.{}".format(frame.name, self.modalities["ground_truth"]["ext"])

            img = Image.open(gt_path)
            img = transform_image(img, self.modalities["ground_truth"]["type"])
            gt_imgs.append(img.unsqueeze(0))

            for mod_idx in range(len(self.modalities.keys()) - 1):
                key = "input_" + str(mod_idx + 1)
                input_path = osp.join(self.data_path, frame.seq_name, self.modalities[key]["name"])
                input_path += "/{}.{}".format(frame.name, self.modalities[key]["ext"])
                img = Image.open(input_path)
                img = transform_image(img, self.modalities[key]["type"])
                input_imgs[i].append(img.unsqueeze(0))

            if frame.next_frame_idx:
                curr_idx = frame.next_frame_idx
            else:
                assert sequence.len - i == 1


        gt_imgs = torch.cat(gt_imgs)
        for i, input_t in enumerate(input_imgs):
            input_imgs[i] = torch.cat(input_t, dim=1)
        input_imgs = torch.cat(input_imgs)
        
        return {
            'gt': gt_imgs.permute(0, 2, 3, 1), 
            'lr': input_imgs.permute(0, 2, 3, 1),
            'seq_idx': sequence.name,
            'frm_idx': frame_idx
        }


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--modalities", type=str, nargs="+", required=True)
    args = parser.parse_args()

    modalities = {
        "ground_truth": {
            "name": args.modalities[0],
            "type": "standard"
        }
    }

    for i, modality in enumerate(args.modalities[1:], start=1):
        modalities["input_" + str(i)] = {
            "name": modality,
            "type": "standard"
        }
    
    # dataset = MultiModalDataset(args.data_path, modalities, tempo_extent=10, crop_size=128)
    # dataloader = DataLoader(dataset, batch_size=2)

    # shapes = False
    # for data in tqdm(dataloader):
    #     if not shapes:
    #         print(data['gt'].shape)
    #         print(data['lr'].shape)
    #         shapes = True
    
    dataset = MultiModalValidationDataset(args.data_path, modalities)
    dataloader = DataLoader(dataset, batch_size=1)

    for data in tqdm(dataloader):
        print(data['gt_tsr'].shape)
        print(data['lr_tsr'].shape)
        print(data['seq_idx'])
        print(f"seq of length {len(data['frame_idx'])}: {data['frame_idx'][:3]}...")
        