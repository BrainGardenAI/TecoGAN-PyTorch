import os
import os.path as osp
import math
import argparse
import yaml
import time
import numpy as np

import torch

from data import create_dataloader, prepare_data, upscale_sequence
from models import define_model
from glob import glob
from models.networks import define_generator
from metrics.metric_calculator import MetricCalculator
from metrics.model_summary import register, profile_model
from utils import base_utils, data_utils
from tqdm import tqdm
from validation import validate


def train(opt):
    # logging
    logger = base_utils.get_logger('base')
    logger.info('{} Options {}'.format('='*20, '='*20))
    base_utils.print_options(opt, logger)

    # create data loader
    train_loader = create_dataloader(opt, dataset_idx='train')

    # create downsampling kernels for BD degradation
    kernel = data_utils.create_kernel(opt)

    # create model
    model = define_model(opt)

    # training configs
    total_sample = len(train_loader.dataset)
    iter_per_epoch = len(train_loader)
    total_iter = opt['train']['total_iter']
    total_epoch = int(math.ceil(total_iter / iter_per_epoch))
    curr_iter = opt['train']['start_iter']

    test_freq = opt['test']['test_freq']
    log_freq = opt['logger']['log_freq']
    ckpt_freq = opt['logger']['ckpt_freq']
    sigma_freq = opt['dataset']['degradation'].get('sigma_freq', 0)
    sigma_inc = opt['dataset']['degradation'].get('sigma_inc', 0)
    sigma_max = opt['dataset']['degradation'].get('sigma_max', 10)

    logger.info('Number of training samples: {}'.format(total_sample))
    logger.info('Total epochs needed: {} for {} iterations'.format(
        total_epoch, total_iter))
    print('device count:', torch.cuda.device_count())
    # train
    for epoch in range(total_epoch):
        for data in tqdm(train_loader):
            # update iter
            curr_iter += 1
            if curr_iter > total_iter:
                logger.info('Finish training')
                break

            # update learning rate
            model.update_learning_rate()

            # prepare data
            data = prepare_data(opt, data, kernel)

            # train for a mini-batch
            model.train(data)

            # update running log
            model.update_running_log()

            # log
            if log_freq > 0 and curr_iter % log_freq == 0:
                # basic info
                msg = '[epoch: {} | iter: {}'.format(epoch, curr_iter)
                for lr_type, lr in model.get_current_learning_rate().items():
                    msg += ' | {}: {:.2e}'.format(lr_type, lr)
                msg += '] '

                # loss info
                log_dict = model.get_running_log()
                msg += ', '.join([
                    '{}: {:.3e}'.format(k, v) for k, v in log_dict.items()])
                if opt['dataset']['degradation']['type'] == 'BD':
                    msg += ' | Sigma: {}'.format(opt['dataset']['degradation']['sigma'])
                logger.info(msg)

            # save model
            if ckpt_freq > 0 and curr_iter % ckpt_freq == 0:
                model.save(curr_iter)

            # evaluate performance
            if test_freq > 0 and curr_iter % test_freq == 0:
                # setup model index
                model_idx = 'G_iter{}'.format(curr_iter)
                if opt['dataset']['degradation']['type'] == 'BD':
                    model_idx = model_idx + str(opt['dataset']['degradation']['sigma'])

                # for each testset
                for dataset_idx in sorted(opt['dataset'].keys()):
                    # use dataset with prefix `test`
                    if not dataset_idx.startswith('validate'):
                        continue
                    validate(opt, model, logger, dataset_idx, model_idx)

        # schedule sigma
        if opt['dataset']['degradation']['type'] == 'BD':
            if sigma_freq > 0 and (epoch + 1) % sigma_freq == 0:
                current_sigma = opt['dataset']['degradation']['sigma']
                opt['dataset']['degradation']['sigma'] = min(current_sigma + sigma_inc, sigma_max)
                kernel = data_utils.create_kernel(opt)
                
                # __getitem__ in custom dataset class uses some crop that depends sigma
                # it is crucial to change this cropsize accordingly if sigma is being changed
                train_loader.dataset.change_cropsize(opt['dataset']['degradation']['sigma'])
                print('kernel changed')
            




def test(opt):
    # logging
    logger = base_utils.get_logger('base')
    if opt['verbose']:
        logger.info('{} Configurations {}'.format('=' * 20, '=' * 20))
        base_utils.print_options(opt, logger)
    # infer and evaluate performance for each model
    for load_path in opt['model']['generator']['load_path_lst']:
        # setup model index
        model_idx = osp.splitext(osp.split(load_path)[-1])[0]
        
        # log
        logger.info('=' * 40)
        logger.info('Testing model: {}'.format(model_idx))
        logger.info('=' * 40)

        # create model
        opt['model']['generator']['load_path'] = load_path
        model = define_model(opt)
        model_idx = osp.basename(opt['model']['generator']['load_path']).split('.')[0]
        # for each test dataset
        for dataset_idx in sorted(opt['dataset'].keys()):
            # use dataset with prefix `test`
            if not dataset_idx.startswith('test'):
                continue
            validate(opt, model, logger, dataset_idx, model_idx, compute_metrics=False)

            logger.info('-' * 40)

    # logging
    logger.info('Finish testing')
    logger.info('=' * 40)


def profile(opt, lr_size, test_speed=False):
    # logging
    logger = base_utils.get_logger('base')
    logger.info('{} Model Information {}'.format('='*20, '='*20))
    base_utils.print_options(opt['model']['generator'], logger)

    # basic configs
    scale = opt['scale']
    device = torch.device(opt['device'])

    # create model
    net_G = define_generator(opt).to(device)

    # get dummy input
    dummy_input_dict = net_G.generate_dummy_input(lr_size)
    for key in dummy_input_dict.keys():
        dummy_input_dict[key] = dummy_input_dict[key].to(device)

    # profile
    register(net_G, dummy_input_dict)
    gflops, params = profile_model(net_G)

    logger.info('-' * 40)
    logger.info('Super-resolute data from {}x{}x{} to {}x{}x{}'.format(
        *lr_size, lr_size[0], lr_size[1]*scale, lr_size[2]*scale))
    logger.info('Parameters (x10^6): {:.3f}'.format(params/1e6))
    logger.info('FLOPs (x10^9): {:.3f}'.format(gflops))
    logger.info('-' * 40)

    # test running speed
    if test_speed:
        n_test = 3
        tot_time = 0

        for i in range(n_test):
            start_time = time.time()
            with torch.no_grad():
                _ = net_G(**dummy_input_dict)
            end_time = time.time()
            tot_time += end_time - start_time

        logger.info('Speed (FPS): {:.3f} (averaged for {} runs)'.format(
            n_test / tot_time, n_test))
        logger.info('-' * 40)


if __name__ == '__main__':
    # ----------------- parse arguments ----------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='directory of the current experiment')
    parser.add_argument('--mode', type=str, required=True,
                        help='which mode to use (train|test|profile)')
    parser.add_argument('--model', type=str, required=True,
                        help='which model to use (FRVSR|TecoGAN)')
    parser.add_argument('--opt', type=str, required=True,
                        help='path to the option yaml file')
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help='GPU index, -1 for CPU')
    parser.add_argument('--lr_size', type=str, default='3x256x256',
                        help='size of the input frame')
    parser.add_argument('--test_speed', action='store_true',
                        help='whether to test the actual running speed')
    args = parser.parse_args()
    
    

    # ----------------- get options ----------------- #
    with open(osp.join(args.exp_dir, args.opt), 'r') as f:
        opt = yaml.load(f.read(), Loader=yaml.FullLoader)


    # ----------------- general configs ----------------- #
    # experiment dir
    opt['exp_dir'] = args.exp_dir

    # random seed
    base_utils.setup_random_seed(opt['manual_seed'])

    # logger
    if args.mode == 'train' or args.mode == 'test':
        if args.mode == 'train':
            logpath = osp.join('results', opt['dataset']['train']['name'], opt['experiment'])
        else:
            logpath = osp.join('results', opt['dataset']['name'], opt['experiment'])
        if not osp.exists(logpath):
            os.mkdir(logpath)
        logpath = logpath + '/train.log'
    base_utils.setup_logger('base', filepath=logpath)
    opt['verbose'] = opt.get('verbose', False)

    # device
    if args.gpu_id >= 0:
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            opt['device'] = 'cuda'
        else:
            opt['device'] = 'cpu'
    else:
        opt['device'] = 'cpu'


    # ----------------- train ----------------- #
    if args.mode == 'train':
        # setup paths
        for dataset_idx in sorted(opt['dataset'].keys()):
            if not (dataset_idx.startswith('test') or dataset_idx.startswith('validate')):
                continue
            if opt['dataset'][dataset_idx]['name'].startswith('Actors'):
                actor_name = opt['dataset'][dataset_idx]['actor_name']
                degradation_type = opt['dataset']['degradation']['type']
                domain_type = opt['dataset'][dataset_idx]['domain']

                gt_segment_folders = 'data/Actors/test/{}/{}'.format(actor_name, domain_type)
                lr_segment_folders = 'data/Actors/test/{}/{}_{}'.format(actor_name, domain_type, degradation_type)

                opt['dataset'][dataset_idx]['gt_seq_dir'] = gt_segment_folders
                opt['dataset'][dataset_idx]['lr_seq_dir'] = lr_segment_folders

        base_utils.setup_paths(opt, mode='train')

        # run
        opt['is_train'] = True
        train(opt)

    # ----------------- test ----------------- #
    elif args.mode == 'test':
        # setup paths
        opt['is_train'] = False

        for dataset_idx in sorted(opt['dataset'].keys()):
            if not (dataset_idx.startswith('test') or dataset_idx.startswith('validate')):
                continue
            if opt['dataset'][dataset_idx]['name'].startswith('Actors'):
                actor_name = opt['dataset'][dataset_idx]['actor_name']
                degradation_type = opt['dataset']['degradation']['type']
                domain_type = opt['dataset'][dataset_idx]['domain']

                gt_segment_folders = 'data/Actors/test/{}/{}'.format(actor_name, domain_type)
                lr_segment_folders = 'data/Actors/test/{}/{}_{}'.format(actor_name, domain_type, degradation_type)

                opt['dataset'][dataset_idx]['gt_seq_dir'] = gt_segment_folders
                opt['dataset'][dataset_idx]['lr_seq_dir'] = lr_segment_folders

        base_utils.setup_paths(opt, mode='test')

        test(opt)

    # ----------------- profile ----------------- #
    elif args.mode == 'profile':
        lr_size = tuple(map(int, args.lr_size.split('x')))

        # run
        profile(opt, lr_size, args.test_speed)

    else:
        raise ValueError(
            'Unrecognized mode: {} (train|test|profile)'.format(args.mode))
