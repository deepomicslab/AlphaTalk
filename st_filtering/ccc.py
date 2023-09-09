import os
import numpy as np
import random
from utils import AlphaTalkCCC, PATHWAY, LR_novel, LR_novel_mouse
from types import SimpleNamespace
import argparse


random.seed(42)
np.random.seed(42)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CCC inference with single-cell ST data.')
    parser.add_argument('--st_file', type=str, default=None, help='file path of the ST data')
    parser.add_argument('--species', type=str, default='Human',
                        help='choose from {Human, Mouse}')
    parser.add_argument('--tmp_dir', type=str, default='./tmp/', help='root directory of the job')
    parser.add_argument('--job_name', type=str, default='job1',
                        help='specify the job name')
    parser.add_argument('--parallel', dest='parallel', action='store_true',
                        help="multiprocessing")
    parser.add_argument("--n_core", type=int, default=4,
                        help="number of processes; only works when parallel is True")
    parser.add_argument('--filtering', type=str, default='mean',
                        help='method for calculating valid expression count threshold; choose from {median, mean, null}')
    parser.add_argument('--dist_thd', dest='dist_thd', action='store_true',
                        help="use the 95 quantiles of distance to filter cell neighbors")
    parser.add_argument('--lrdb', dest='lrdb', action='store_true',
                        help="use LRIs from LRDB as well")

    parser.set_defaults(parallel=False)
    parser.set_defaults(dist_thd=True)
    parser.set_defaults(lrdb=True)

    args = parser.parse_args()

    # data path
    data_path = {
        'st_file': args.st_file,
        'pathway': PATHWAY
    }
    if args.species == 'Human':
        data_path['lr_file'] = LR_novel
    elif args.species == 'Mouse':
        data_path['lr_file'] = LR_novel_mouse
    else:
        raise Exception('Unknown species name')
    data_path = SimpleNamespace(**data_path)

    atalk = AlphaTalkCCC(args=args, input_data=data_path)
    atalk.run()

