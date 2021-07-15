import os
import os.path as osp
import random
import logging

import numpy as np
import torch
import torch.nn.functional as F
import cv2


def setup_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(name, filepath=''):
    if filepath != '':
        logging.basicConfig(
            filename=filepath,
            filemode='a',
            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S',
            level=logging.INFO #logging.DEBUG
        )
    # create a logger
    base_logger = logging.getLogger(name=name)
    base_logger.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter(fmt='%(asctime)s [%(levelname)s]: %(message)s')
    # create a stream handler & set format
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    # add handlers
    base_logger.addHandler(sh)


def get_logger(name):
    logger = logging.getLogger(name)
    return logger


def print_options(opt, logger, tab=''):
    for key, val in opt.items():
        if isinstance(val, dict):
            logger.info('{}{}:'.format(tab, key))
            print_options(val, logger, tab + '  ')
        else:
            logger.info('{}{}: {}'.format(tab, key, val))


def retrieve_files(dir, suffix='png|jpg'):
    """ retrive files with specific suffix under dir and sub-dirs recursively
    """

    def retrieve_files_recursively(dir, file_lst):
        for d in sorted(os.listdir(dir)):
            dd = osp.join(dir, d)

            if osp.isdir(dd):
                retrieve_files_recursively(dd, file_lst)
            else:
                if osp.splitext(d)[-1].lower() in ['.' + s for s in suffix]:
                    file_lst.append(dd)

    if not dir:
        return []

    if isinstance(suffix, str):
        suffix = suffix.split('|')

    file_lst = []
    retrieve_files_recursively(dir, file_lst)
    file_lst.sort()

    return file_lst


def setup_paths(opt, mode):

    def setup_ckpt_dir():
        ckpt_dir = opt['train'].get('ckpt_dir')
        if not ckpt_dir:
            # use default dir
            ckpt_dir = osp.join(opt['exp_dir'], 'train', 'ckpt')
            opt['train']['ckpt_dir'] = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)

    def setup_res_dir():
        res_dir = opt['test'].get('res_dir')
        if not res_dir:
            # use default dir
            res_dir = osp.join(opt['exp_dir'], 'test', 'results')
            opt['test']['res_dir'] = res_dir
        os.makedirs(res_dir, exist_ok=True)

    def setup_json_path():
        json_dir = opt['test'].get('json_dir')
        if not json_dir:
            # use default dir
            json_dir = osp.join(opt['exp_dir'], 'test', 'metrics')
            opt['test']['json_dir'] = json_dir
        os.makedirs(json_dir, exist_ok=True)

    def setup_model_path():
        load_path = opt['model']['generator'].get('load_path')
        if not load_path:
            raise ValueError('Generator path needs to be specified for testing')

        # parse models
        ckpt_dir, model_idx = osp.split(load_path)
        model_idx = osp.splitext(model_idx)[0]
        if model_idx == '*':
            # test a serial of models  TODO: check validity
            start_iter = opt['test']['start_iter']
            end_iter = opt['test']['end_iter']
            freq = opt['test']['test_freq']
            opt['model']['generator']['load_path_lst'] = [
                osp.join(ckpt_dir, 'G_iter{}.pth'.format(iter))
                for iter in range(start_iter, end_iter + 1, freq)]
        else:
            # test a single model
            opt['model']['generator']['load_path_lst'] = [
                osp.join(ckpt_dir, '{}.pth'.format(model_idx))]

    if mode == 'train':
        setup_ckpt_dir()

        for dataset_idx in opt['dataset'].keys():
            if not dataset_idx.startswith('test'):
                continue

            if opt['test'].get('save_res'):
                setup_res_dir()

            if opt['test'].get('save_json'):
                setup_json_path()

    elif mode == 'test':
        setup_model_path()

        for dataset_idx in opt['dataset'].keys():
            if not dataset_idx.startswith('test'):
                continue

            if opt['test'].get('save_res'):
                setup_res_dir()

            if opt['test'].get('save_json'):
                setup_json_path(dataset_idx)


def resize(image, downsample_factor=4):
    h, w = image.shape[-2:]
    h, w = recompute_hw(h, w, downsample_factor)
    return F.interpolate(image, size=(h, w))


def recompute_hw(h, w, factor):
    if downsample_depth(h) < factor:
        h = nearest_power_of_two(h)
    if downsample_depth(w) < factor:
        w = nearest_power_of_two(w)
    return h, w


def downsample_depth(number):
    current_depth = 0
    current_number = number
    while current_number % 2 == 0:
        current_depth += 1
        current_number //= 2
    return current_depth


def nearest_power_of_two(number):
    power = 0
    previous = 1
    current = 1
    while current < number:
        previous = current
        current *= 2
    
    if abs(current - number) < abs(previous - number):
        return current
    
    return previous