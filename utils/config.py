import argparse
from dataset.scannet import ScanNetDataset
from dataset.matterport import MatterportDataset
from dataset.scannetpp import ScanNetPPDataset
from dataset.demo import DemoDataset
import json

def update_args(args):
    config_path = f'configs/{args.config}.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    for key in config:
        setattr(args, key, config[key])
    return args

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_name', type=str)
    parser.add_argument('--seq_name_list', type=str)
    parser.add_argument('--config', type=str, default='scannet')
    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()
    args = update_args(args)
    return args

def get_dataset(args):
    if args.dataset == 'scannet':
        dataset = ScanNetDataset(args.seq_name)
    elif args.dataset == 'scannetpp':
        dataset = ScanNetPPDataset(args.seq_name)
    elif args.dataset == 'matterport3d':
        dataset = MatterportDataset(args.seq_name)
    elif args.dataset == 'demo':
        dataset = DemoDataset(args.seq_name)
    else:
        print(args.dataset)
        raise NotImplementedError
    return dataset

