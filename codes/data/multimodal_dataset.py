from numpy.lib.arraysetops import isin
import torch
import numpy as np
import random

import cv2
import os
from os import path as osp, stat

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple, Dict

from .transforms.torch_transforms import transform_image


# TODO: replace with Gosha's version (with ext as str)
def get_ext(file_path_wo_ext, ext_list):
    for ext in ext_list:
        if os.path.exists(file_path_wo_ext+ext):
            return ext
    return None

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
    def __init__(self, data_path: str, modalities: Dict, tempo_extent: int, crop_size: int,
                 add_background: bool = False):
        super().__init__()

        self.data_path = data_path 
        self.modalities = modalities
        self.tempo_extent = tempo_extent
        self.add_background = add_background
        self.crop_size = crop_size
        self.frame_list, self.seq_list = self._build_frame_and_seq_list(data_path, modalities)
        
        # # getting width and height of the samples
        # frm = self.frame_list[0]
        # frm_path = osp.join(data_path, frm.seq_name, modalities["ground_truth"]["name"], frm.name)
        # if isinstance(modalities["ground_truth"]["ext"], str):
        #     frm_path += ".{}".format(modalities["ground_truth"]["ext"])
        # else:
        #     frm_path += get_ext(file_path_wo_ext=frm_path, ext_list=modalities["ground_truth"]["ext"])
        #
        # img = Image.open(frm_path)
        # self._w, self._h = img.size

    def __len__(self):
        return len(self.frame_list)
    
    def read_frame(self, frame: Frame, bbox: Tuple[int] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        gt_path = osp.join(self.data_path, frame.seq_name, self.modalities["ground_truth"]["name"], frame.name)
        if isinstance(self.modalities["ground_truth"]["ext"], str):
            gt_path += ".{}".format(self.modalities["ground_truth"]["ext"])
        else:
            gt_path += get_ext(file_path_wo_ext=gt_path, ext_list=self.modalities["ground_truth"]["ext"])

        gt_img = Image.open(gt_path)

        gt_img_bg = None
        seg_img_bg = None
        if self.add_background:
            seg_path = osp.join(self.data_path, frame.seq_name, "segments", frame.name + ".png")
            gt_img_bg = cv2.imread(gt_path)
            seg_img_bg = cv2.cvtColor(cv2.imread(seg_path), cv2.COLOR_BGR2GRAY).astype(np.bool)
            gt_img_bg = cv2.bitwise_and(gt_img_bg, gt_img_bg, mask=np.bitwise_not(seg_img_bg).astype(np.uint8))

            if bbox:
                gt_img_bg = gt_img_bg[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                seg_img_bg = seg_img_bg[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        gt_img = transform_image(gt_img, self.modalities["ground_truth"]["type"])

        input_imgs = []
        for mod_idx in range(len(self.modalities.keys()) - 1):
            key = "input_" + str(mod_idx + 1)

            input_path = osp.join(self.data_path, frame.seq_name, self.modalities[key]["name"], frame.name)
            if isinstance(self.modalities["ground_truth"]["ext"], str):
                input_path += ".{}".format(self.modalities["ground_truth"]["ext"])
            else:
                input_path += get_ext(file_path_wo_ext=input_path, ext_list=self.modalities["ground_truth"]["ext"])

            img = Image.open(input_path)
            if bbox:
                img = img.crop(bbox)
            img = transform_image(img, self.modalities[key]["type"], background=gt_img_bg, mask=seg_img_bg)

            # we only need to use them with first input image
            gt_img_bg, seg_img_bg = None, None

            input_imgs.append(img)
        
        return gt_img, input_imgs
        
    def read_sequence(self, frame_idx: int, seq_len: int) -> Tuple[List, List, List]:
        gt_imgs = []
        input_imgs = [[] for _ in range(seq_len)]
        frame_indices= []
        curr_idx = frame_idx
        forward = True

        # getting width and height of the samples
        frm = self.frame_list[frame_idx]
        frm_path = osp.join(self.data_path, frm.seq_name, self.modalities["ground_truth"]["name"], frm.name)
        if isinstance(self.modalities["ground_truth"]["ext"], str):
            frm_path += ".{}".format(self.modalities["ground_truth"]["ext"])
        else:
            frm_path += get_ext(file_path_wo_ext=frm_path, ext_list=self.modalities["ground_truth"]["ext"])
        img = Image.open(frm_path)
        _w, _h = img.size
        bbox = self.get_random_crop(self.crop_size, _w, _h)
        
        for i in range(seq_len):
            frame = self.frame_list[curr_idx]
            gt_img, mod_imgs = self.read_frame(frame, bbox)
            for img in mod_imgs:
                input_imgs[i].append(img.unsqueeze(0))
            frame_indices.append(frame.name)
            gt_imgs.append(gt_img.unsqueeze(0))

            if frame.next_frame_idx and forward:
                curr_idx = frame.next_frame_idx
            else:
                forward = False
                curr_idx = frame.prev_frame_idx
        
        return gt_imgs, input_imgs, frame_indices
    
    def __getitem__(self, idx: int) -> Dict:
        #bbox = self.get_random_crop(self.crop_size, self._w, self._h)
        #gt_imgs, input_imgs, _ = self.read_sequence(idx, self.tempo_extent, bbox)
        gt_imgs, input_imgs, _ = self.read_sequence(idx, self.tempo_extent)

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
            
            frame_start_idx = len(frame_list)
            num_frames = 0
            for frame_name in frame_names:
                frame_name = frame_name.split('.')[0]
                if not self._check_all_modalities(frame_name, modalities, data_path, seq_name):
                    continue
                num_frames += 1
                curr_frame = Frame(frame_name, seq_name, None, None)
                frame_list.append(curr_frame)
                if prev_frame:
                    prev_frame.next_frame_idx = len(frame_list) - 1
                    curr_frame.prev_frame_idx = len(frame_list) - 2
                prev_frame = curr_frame    

            seq_list.append(Sequence(seq_name, frame_start_idx, num_frames))

        return frame_list, seq_list
    
    @staticmethod
    def get_random_crop(crop_size, image_w, image_h):
        left = random.randint(0, image_w - crop_size)
        upper = random.randint(0, image_h - crop_size)
        return left, upper, left + crop_size, upper + crop_size
    
    def _check_all_modalities(self, frame_name: str, modalities: Dict, data_path: str, seq_name: str) -> bool:
        for key in modalities:
            path = osp.join(data_path, seq_name, modalities[key]["name"], frame_name)
            if isinstance(modalities["ground_truth"]["ext"], str):
                path += ".{}".format(modalities["ground_truth"]["ext"])
            else:
                ext = get_ext(file_path_wo_ext=path, ext_list=modalities["ground_truth"]["ext"])
                if not ext:
                    return False
                path += ext
            if not osp.exists(path):
                return False
        if self.add_background:
            seg_path = osp.join(data_path, seq_name, "segments", frame_name + ".png")
            if not osp.exists(seg_path):
                return False
        return True


class MultiModalValidationDataset(MultiModalDataset):
    def __init__(self, data_path: str, modalities: Dict, framewise=False, add_background: bool = False):
        super().__init__(data_path, modalities, None, None, add_background=add_background)
        self.framewise = framewise
    
    def __len__(self):
        return len(self.seq_list)
    
    def __getitem__(self, idx):
        sequence = self.seq_list[idx]
        curr_idx = sequence.first_frame

        if self.framewise:
            return {
                'sequence_gen': self.get_sequence_generator(curr_idx, sequence.len),
                'seq_idx': sequence.name
            }
        
        gt_imgs, input_imgs, frame_idx = self.read_sequence(curr_idx, sequence.len)

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
    
    def get_sequence_generator(self, frame_idx, seq_len):
        curr_idx = frame_idx

        for i in range(seq_len):
            frame = self.frame_list[curr_idx]
            gt_img, input_imgs = self.read_frame(frame)
            input_imgs = torch.cat([img.unsqueeze(0) for img in input_imgs], dim=1)

            yield {
                'gt': gt_img.unsqueeze(0),
                'lr': input_imgs,
                'frm_idx': frame.name
            }

            if frame.next_frame_idx:
                curr_idx = frame.next_frame_idx
            else:
                assert seq_len - i == 1


class MultiModalValidationLoader:
    def __init__(self, dataset: MultiModalValidationDataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        for idx in range(len(self.dataset)):
            yield self.dataset[idx]


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
    
    dataset = MultiModalDataset(args.data_path, modalities, tempo_extent=10, crop_size=128)
    dataloader = DataLoader(dataset, batch_size=2)

    shapes = False
    for data in tqdm(dataloader):
        if not shapes:
            print(data['gt'].shape)
            print(data['lr'].shape)
            shapes = True
    
    # dataset = MultiModalValidationDataset(args.data_path, modalities)
    # dataloader = DataLoader(dataset, batch_size=1)
    #
    # for data in tqdm(dataloader):
    #     print(data['gt_tsr'].shape)
    #     print(data['lr_tsr'].shape)
    #     print(data['seq_idx'])
    #     print(f"seq of length {len(data['frame_idx'])}: {data['frame_idx'][:3]}...")
        