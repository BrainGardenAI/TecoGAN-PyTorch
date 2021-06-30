import torch
import numpy as np
import torch.nn.functional as F
from os import path as osp
from torch.utils import data
from torch.utils.data import DataLoader

from .paired_lmdb_dataset import PairedLMDBDataset
from .unpaired_lmdb_dataset import UnpairedLMDBDataset
from .paired_folder_dataset import PairedFolderDataset
from .simple_dataset import SimpleDataset
from .dataset_for_validation import ValidationDataset
from .multimodal_dataset import MultiModalDataset, MultiModalValidationDataset, MultiModalValidationLoader
from utils.data_utils import float32_to_uint8


def create_dataloader(opt, dataset_idx='train'):
    # setup params
    data_opt = opt['dataset'].get(dataset_idx)
    degradation_type = opt['dataset']['degradation']['type']

    # -------------- loader for training -------------- #
    if dataset_idx == 'train':
        # check dataset
        # assert data_opt['name'] in ('VimeoTecoGAN', 'VimeoTecoGAN-sub', 'Actors'), \
        #     'Unknown Dataset: {}'.format(data_opt['name'])
        if degradation_type == 'Multimodal':
            dataset = MultiModalDataset(
                osp.join(data_opt['data_path'], data_opt['domain']),
                data_opt['modalities'],
                opt['train']['tempo_extent'],
                data_opt['gt_crop_size']
            )

        elif degradation_type == 'BI' or degradation_type == 'Style':
            # create dataset
            dataset = PairedLMDBDataset(
                data_opt,
                scale=opt['scale'],
                tempo_extent=opt['train']['tempo_extent'],
                moving_first_frame=opt['train'].get('moving_first_frame', False),
                moving_factor=opt['train'].get('moving_factor', 1.0))

        elif degradation_type == 'BD':
            # enlarge crop size to incorporate border size
            sigma = opt['dataset']['degradation']['sigma']
            enlarged_crop_size = data_opt['crop_size'] + 2 * int(sigma * 3.0)

            # create dataset
            dataset = UnpairedLMDBDataset(
                data_opt,
                crop_size=enlarged_crop_size,  # override
                original_crop_size=data_opt['crop_size'],
                tempo_extent=opt['train']['tempo_extent'],
                moving_first_frame=opt['train'].get('moving_first_frame', False),
                moving_factor=opt['train'].get('moving_factor', 1.0))

        else:
            raise ValueError('Unrecognized degradation type: {}'.format(
                degradation_type))

        # create data loader
        loader = DataLoader(
            dataset=dataset,
            batch_size=data_opt['batch_size'],
            shuffle=True,
            num_workers=data_opt['num_workers'],
            pin_memory=data_opt['pin_memory'])

    # ------------- loader for validation ------------- #
    elif dataset_idx.startswith('validate_BD'):
        loader = DataLoader(
            dataset=ValidationDataset(
                data_opt,
                scale=opt['scale'],
                sigma=opt['dataset']['degradation'].get('sigma', 1.5),
                data_preparation_method=prepare_data
            ),
            batch_size=1,
            shuffle=False,
            num_workers=data_opt['num_workers'],
            pin_memory=data_opt['pin_memory']
        )

    # -------------- loader for testing -------------- #
    elif dataset_idx.startswith('test') or dataset_idx.startswith('validate'):
        if data_opt['name'] == 'Multimodal':
            dataset = MultiModalValidationDataset(
                osp.join(data_opt['data_path'], data_opt['domain']),
                data_opt['modalities'],
                data_opt['framewise']
            )
            if data_opt['framewise']:
                return MultiModalValidationLoader(dataset)
        else:
            dataset = PairedFolderDataset(data_opt, scale=opt['scale'])
        # create data loader
        loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=data_opt['num_workers'],
            pin_memory=data_opt['pin_memory']
        )

    # ------------- loader for getting lr images ------------- #
    elif dataset_idx.startswith('all'):
        dataset = SimpleDataset(data_opt, scale=opt['scale'])
        # create data loader
        loader = DataLoader(
            dataset=SimpleDataset(
                data_opt, 
                scale=opt['scale']
            ),
            batch_size=1,
            shuffle=False,
            num_workers=data_opt['num_workers'],
            pin_memory=data_opt['pin_memory']
        )

    else:
        raise ValueError('Unrecognized dataset index: {}'.format(dataset_idx))

    return loader


def apply_BD_iteratively(data, kernel, filter_size, scale, device, batch_size):
    gt_data = data['gt']
    n, t, c, gt_h, gt_w = gt_data.size()
    lr_h = (gt_h - filter_size) // scale + 1
    lr_w = (gt_w - filter_size) // scale + 1
    lr_data = []
    gt_data = gt_data.view(n * t, c, gt_h, gt_w)
    for idx_start in range(0, n * t, batch_size):
        idx_end = min(idx_start + batch_size, n * t)
        data_to_process = gt_data[idx_start : idx_end].to(device)
        lr_data_item = F.conv2d(
            data_to_process, kernel, stride=scale, bias=None, padding=0
        )
        lr_data.append(lr_data_item.unsqueeze(0))

    lr_data = torch.cat(lr_data).view(n, t, c, lr_h, lr_w)

    return lr_data
    

def apply_BD_at_once(data, kernel, filter_size, scale, device):
    gt_with_border = data['gt'].to(device)
    n, t, c, gt_h, gt_w = gt_with_border.size()
    lr_h = (gt_h - filter_size) // scale + 1
    lr_w = (gt_w - filter_size) // scale + 1

    # generate lr data
    gt_with_border = gt_with_border.view(n * t, c, gt_h, gt_w)
    lr_data = F.conv2d(
        gt_with_border, kernel, stride=scale, bias=None, padding=0)
    lr_data = lr_data.view(n, t, c, lr_h, lr_w)

    return lr_data


def upscale_sequence(data, gt_h, gt_w, batch_size=10):
    data = data.permute(0, 3, 1, 2)
    t, c, h, w = data.size()
    result = []
    for idx_start in range(0, t, batch_size):
        idx_end = min(idx_start + batch_size, t)
        data_item = data[idx_start : idx_end]
        data_item = F.interpolate(data_item, size=(gt_h, gt_w), mode='bilinear', align_corners=False)
        result.append(data_item.cpu().numpy())
    result = np.concatenate(result)
    result = result.transpose(0, 2, 3, 1)
    result = float32_to_uint8(result)
    return result


def prepare_data(opt, data, kernel, batch_size=-1, return_gt_data=True):
    """ prepare gt, lr data for training

        for BD degradation, generate lr data and remove border of gt data
        for BI degradation, return data directly

    """
    device = torch.device(opt['device'])
    degradation_type = opt['dataset']['degradation']['type']

    if not degradation_type == 'BD':
        gt_data, lr_data = data['gt'].to(device), data['lr'].to(device)

    elif degradation_type == 'BD':
        # setup params
        scale = opt['scale']
        sigma = opt['dataset']['degradation'].get('sigma', 1.5)
        # border_size = 0 # int(sigma * 3.0)
        filter_size = kernel.shape[-1]
        border_size = (filter_size - 1) // 2

        if batch_size != -1:
            lr_data = apply_BD_iteratively(data, kernel, filter_size, scale, device, batch_size)
        else:
            lr_data = apply_BD_at_once(data, kernel, filter_size, scale, device)

        n, t, c, lr_h, lr_w = lr_data.size()

        # remove gt border
        gt_data = data['gt'][
            ...,
            border_size: border_size + scale * lr_h,
            border_size: border_size + scale * lr_w
        ]
        gt_data = gt_data.view(n, t, c, scale * lr_h, scale * lr_w)

        if return_gt_data:
            return {'gt': gt_data.to(device), 'lr': lr_data}
        else:
            return {'lr': lr_data}

    else:
        raise ValueError('Unrecognized degradation type: {}'.format(
            degradation_type))

    return { 'gt': gt_data, 'lr': lr_data }
