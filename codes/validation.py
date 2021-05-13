import os
import os.path as osp
import numpy as np
import torch

from data import create_dataloader, prepare_data, upscale_sequence
from metrics.metric_calculator import MetricCalculator
from metrics.model_summary import register, profile_model
from utils import base_utils, data_utils
from tqdm import tqdm


def validate(opt, model, logger, dataset_idx, model_idx):
    ds_name = opt['dataset'][dataset_idx]['name']
    if ds_name == 'Actors':
        actor_name = opt['dataset'][dataset_idx]['actor_name']
        domain_type = opt['dataset'][dataset_idx]['domain']
    logger.info(
        'Testing on {}: {}'.format(dataset_idx, ds_name))
    
    # create data loader
    test_loader = create_dataloader(opt, dataset_idx=dataset_idx)
    if not len(test_loader.dataset):
        return
    
     # define metric calculator
    metric_calculator = MetricCalculator(opt)

    # infer and compute metrics for each sequence
                    
    for data in tqdm(test_loader):

        input_data_type = opt['dataset']['degradation']['type']
        input_seq, output_seq, seq_idx, frm_idx = data_processing(model, data, test_loader, input_data_type)

        seq_to_save = np.dstack([output_seq, input_seq]) # t.h.2w.c|rgb|uint8

        # save results (optional)
        if opt['test']['save_res']:
            res_dir = osp.join(
                opt['test']['res_dir'], 
                ds_name, 
                opt['dataset']['degradation']['type'], 
                opt['experiment'], 
                actor_name, 
                domain_type, 
                model_idx
            )
            res_seq_dir = osp.join(res_dir, seq_idx)
            data_utils.save_sequence(
                res_seq_dir, seq_to_save, frm_idx, to_bgr=True)

        # compute metrics for the current sequence
        true_seq_dir = osp.join(
            opt['dataset'][dataset_idx]['gt_seq_dir'], seq_idx)
        metric_calculator.compute_sequence_metrics(
            seq_idx, true_seq_dir, '', pred_seq=output_seq)

    # save/print metrics
    if opt['test'].get('save_json'):
        # save results to json file
        json_path = osp.join(
            opt['test']['json_dir'], '{}_avg.json'.format(ds_name))
        metric_calculator.save_results(
            model_idx, json_path, override=True)
    else:
        # print directly
        metric_calculator.display_results()


def data_processing(model, data, test_loader, input_data_type):
    if input_data_type == 'BD':
        lr_data = test_loader.dataset.apply_BD(data['gt'])['lr'][0]
    elif input_data_type == 'BI' or input_data_type == 'Style':
        lr_data = data['lr'][0]

    seq_idx = data['seq_idx'][0]
    frm_idx = [frm_idx[0] for frm_idx in data['frm_idx']]
    output_seq = model.infer(lr_data)  # thwc|rgb|uint8
    _, h, w, _ = output_seq.shape
    
    if input_data_type != 'Style':
        input_seq = upscale_sequence(lr_data, h, w) # thwc|rgb|uint8
    else:
        input_seq = lr_data.detach().cpu().numpy()
        input_seq = data_utils.float32_to_uint8(input_seq)
    return input_seq, output_seq, seq_idx, frm_idx
