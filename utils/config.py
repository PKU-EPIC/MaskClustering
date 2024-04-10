import argparse
from dataset.scannet import ScanNetDataset
from dataset.matterport import MatterportDataset
from dataset.scannetpp import ScanNetPPDataset
import json

def update_args(args):
    config_path = f'data/configs/{args.config_type}.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    for key in config:
        setattr(args, key, config[key])
    return args

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', default='scannet', type=str)
    parser.add_argument('--seq_name', type=str)
    parser.add_argument('--step', default=10, type=int)
    parser.add_argument('--seq_name_list', type=str)
    parser.add_argument('--config_type', type=str, default='best_2')
    parser.add_argument('--mask_generator', type=str, default='cropformer_intrinsics')
    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()
    args = update_args(args)
    return args

def get_dataset(args):
    if args.dataset_type == 'scannet':
        dataset = ScanNetDataset(args.seq_name, args.step, args.mask_generator)
    elif args.dataset_type == 'scannetpp':
        dataset = ScanNetPPDataset(args.seq_name, args.step)
    elif args.dataset_type == 'matterport3d':
        dataset = MatterportDataset(args.seq_name, 'cropformer')
    else:
        print(args.dataset_type)
        raise NotImplementedError
    return dataset

