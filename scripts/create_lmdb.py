import os
import os.path as osp
import argparse
import glob
import lmdb
import pickle
import random

import cv2
import numpy as np

from tqdm import tqdm


def create_lmdb(dataset, raw_dir, lmdb_dir, filter_file=''):
    assert dataset in ('VimeoTecoGAN', 'VimeoTecoGAN-sub', 'Actors'), \
        'Unknown Dataset: {}'.format(dataset)
    print('Creating lmdb dataset: {}'.format(dataset))

    # retrieve sequences
    if filter_file:  # dump selective data into lmdb
        with open(filter_file, 'r') as f:
            seq_dir_lst = sorted([line.strip() for line in f])
    else:
        seq_dir_lst = [sorted(os.listdir(rd)) for rd in raw_dir]
    
    counter = 0
    for i, _ in enumerate(raw_dir):
        counter += len(seq_dir_lst[i])
    print('Number of sequences: {}'.format(counter))
    # compute space to allocate
    print(raw_dir, seq_dir_lst)
    print('Calculating space needed for LMDB generation ... ', end='')
    nbytes = 0
    extension = 'jpg' if dataset == 'Actors' else 'png'
    for i, rd in enumerate(raw_dir):
        for seq_dir in seq_dir_lst[i]:
            if dataset == 'Actors':
                seq_dir = seq_dir + '/frames'
            frm_path_lst = sorted(glob.glob(osp.join(rd, seq_dir, ('*.' + extension))))
            nbytes_per_frm = cv2.imread(frm_path_lst[0], cv2.IMREAD_UNCHANGED).nbytes
            nbytes += len(frm_path_lst)*nbytes_per_frm
    alloc_size = round(1.5*nbytes)
    print('{:.2f} GB'.format(alloc_size / (1 << 30)))

    # create lmdb environment
    env = lmdb.open(lmdb_dir, map_size=alloc_size)

    # write data to lmdb
    commit_freq = 100
    keys = []
    txn = env.begin(write=True)
    for i, rd in enumerate(raw_dir):
        for b, seq_dir in enumerate(seq_dir_lst[i]):
            if dataset == 'Actors':
                seq_dir = seq_dir + '/frames'
            # log
            print('Processing {} ({}/{})\r'.format(
                seq_dir, b, len(seq_dir_lst)), end='')

            # setup
            frm_path_lst = sorted(glob.glob(osp.join(rd, seq_dir, ('*.' + extension))))
            n_frm = len(frm_path_lst)

            # read frames
            for i in tqdm(range(n_frm)):
                frm = cv2.imread(frm_path_lst[i], cv2.IMREAD_UNCHANGED)
                frm = np.ascontiguousarray(frm[..., ::-1])  # hwc|rgb|uint8

                h, w, c = frm.shape
                key = '{}_{}x{}x{}_{:04d}'.format(seq_dir, n_frm, h, w, i)
                txn.put(key.encode('ascii'), frm)
                keys.append(key)

            # commit
                if i % commit_freq == 0:
                    txn.commit()
                    txn = env.begin(write=True)

    txn.commit()
    env.close()

    # create meta information
    meta_info = {
        'name': dataset,
        'keys': keys
    }
    pickle.dump(meta_info, open(osp.join(lmdb_dir, 'meta_info.pkl'), 'wb'))


def check_lmdb(dataset, lmdb_dir, display=True):
    extension = 'jpg' if dataset == 'Actors' else 'png'
    def visualize(win, img):
        if display:
            cv2.namedWindow(win, 0)
            cv2.resizeWindow(win, img.shape[-2], img.shape[-3])
            cv2.imshow(win, img[..., ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.imwrite('_'.join(win.split('/')) + '.' + extension, img[..., ::-1])

    assert dataset in ('VimeoTecoGAN', 'VimeoTecoGAN-sub', 'Actors'), \
        'Unknown Dataset: {}'.format(dataset)
    print('Checking lmdb dataset: {}'.format(dataset))

    # load keys
    meta_info = pickle.load(open(osp.join(lmdb_dir, 'meta_info.pkl'), 'rb'))
    keys = meta_info['keys']
    print('Number of keys: {}'.format(len(keys)))

    # randomly select frame to visualize
    with lmdb.open(lmdb_dir) as env:
        for i in range(3):  # replace to whatever you want
            idx = random.randint(0, len(keys) - 1)
            key = keys[idx]

            # parse key
            key_lst = key.split('_')
            vid, sz, frm = '_'.join(key_lst[:-2]), key_lst[-2], key_lst[-1]
            sz = tuple(map(int, sz.split('x')))
            sz = (*sz[1:], 3)
            print('video index: {} | size: {} | # of frame: {}'.format(
                vid, sz, frm))

            with env.begin() as txn:
                buf = txn.get(key.encode('ascii'))
                val = np.frombuffer(buf, dtype=np.uint8).reshape(*sz) # HWC

            visualize(key, val)


if __name__ == '__main__':
    # parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='VimeoTecoGAN or VimeoTecoGAN-sub')
    parser.add_argument('--data_type', type=str, required=True,
                        help='GT or Bicubic4xLR')
    parser.add_argument('--actors', nargs='+', type=str,
                        help="list of actors to process")
    args = parser.parse_args()

    # setup
    if args.dataset == 'Actors':
        if len(args.actors) == 1 and args.actors[0] == '__ALL__':
            path = 'data/{}/train/*/{}/'.format(args.dataset, args.data_type)
            raw_dir_list = [glob.glob(path)]
            lmdb_dir_list = ['data/{}/train/{}.lmdb'.format(args.dataset, args.data_type)]
            filter_file_list = ['']
        else:
            if not len(args.actors):
                actor_list = glob(data_path + '/*/')
                actor_list = [osp.basename(actor) for actor in args.actors]

            raw_dir_list = [['data/{}/train/{}/{}'.format(args.dataset, actor, args.data_type)] for actor in args.actors]
            lmdb_dir_list = ['data/{}/train/{}/{}.lmdb'.format(args.dataset, actor, args.data_type) for actor in args.actors]
            filter_file_list = ['' for _ in range(len(args.actors))]
    else:
        raw_dir_list = [['data/{}/{}'.format(args.dataset, args.data_type)]]
        lmdb_dir_list = [['data/{}/{}.lmdb'.format(args.dataset, args.data_type)]]
        filter_file_list = ['']

    # run
    for raw_dir, lmdb_dir, filter_file in zip(raw_dir_list, lmdb_dir_list, filter_file_list):
        if osp.exists(lmdb_dir):
            print('Dataset [{}] already exists'.format(args.dataset))
            print('Checking the LMDB dataset ...')
            check_lmdb(args.dataset, lmdb_dir)
        else:
            create_lmdb(args.dataset, raw_dir, lmdb_dir, filter_file)

            print('Checking the LMDB dataset ...')
            check_lmdb(args.dataset, lmdb_dir)

