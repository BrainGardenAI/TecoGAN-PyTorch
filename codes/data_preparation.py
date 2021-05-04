import os.path as osp
import numpy as np
import argparse
import yaml
import glob
import cv2
import os

from data import create_dataloader, prepare_data
from utils import data_utils
from tqdm import tqdm


def downscale_data(opt):
    for dataset_idx in sorted(opt['dataset'].keys()):
        if not dataset_idx.startswith('all'):
            continue
        print(opt['dataset'][dataset_idx]['gt_seq_dir'])
        print(opt['dataset'][dataset_idx]['segment'])
        loader = create_dataloader(opt, dataset_idx=dataset_idx)
        degradation_type = opt['dataset']['degradation']['type']
        if degradation_type == 'BD':
            kernel = data_utils.create_kernel(opt)
    
        for item in tqdm(loader, ascii=True):
            if degradation_type == 'BD':
                data = prepare_data(opt, item, kernel)
            else:
                data = data_utils.BI_downsample(opt, item)
            lr_data = data['lr']
            gt_data = data['gt']
            img = lr_data.squeeze(0).squeeze(0).permute(1, 2, 0).cpu().numpy()

            path = osp.join(
                'data', 
                opt['dataset']['common']['name'],
                'test',
                opt['dataset']['actor_name'], 
                opt['data_type'] + '_' + opt['dataset']['degradation']['type'],
                opt['dataset'][dataset_idx]['segment'],
                'frames'
            )
            os.makedirs(path, exist_ok=True)
            path = osp.join(path, item['frame_key'][0])
            img = img * 255.0
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            cv2.imwrite(path, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp-dir', type=str, required=True,
                        help='directory of the current experiment')
    parser.add_argument('--opt', type=str, required=True,
                        help='path to the option yaml file')
    parser.add_argument('--actor', type=str, required=True,
                        help='Name of an actor to process')
    parser.add_argument('--data-type', type=str, required=True,
                        help='Domain type: real or virtual')
    parser.add_argument('--degradation-type', type=str, required=True,
                        help='Type of image downscaling, example: BD for blur downscaling')
    parser.add_argument('--sigma', type=float, required=False, default=1.5,
                        help='Sigma parameter of blur downscaling')

    args = parser.parse_args()

    with open(osp.join(args.exp_dir, args.opt), 'r') as f:
        opt = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    opt['exp_dir'] = args.exp_dir
    opt['dataset']['degradation']['type'] = args.degradation_type
    opt['dataset']['degradation']['sigma'] = args.sigma
    opt['dataset']['actor_name'] = args.actor
    opt['data_type'] = args.data_type
    path_to_segment_folders = 'data/Actors/test/{}/{}'.format(args.actor, args.data_type)
    segments = glob.glob(path_to_segment_folders + '/*/')
    print(segments)
    common = opt['dataset']['common']
    for num, s in enumerate(segments):
        opt['dataset']['all' + str(num+1)] = common
        opt['dataset']['all' + str(num+1)]['gt_seq_dir'] = s
        opt['dataset']['all' + str(num+1)]['segment'] = osp.basename(s[:-1])
    
    opt['device'] = 'cuda' 
    downscale_data(opt)
    print('Done.')
