'''
    Clean all output files produced by this project.
'''

from utils.config import get_dataset, get_args
import os
from tqdm import tqdm

def delete_files(dataset):
    dirs_to_be_deleted = [f'{dataset.root}/output']
    for dir_to_be_deleted in dirs_to_be_deleted:
        if os.path.exists(dir_to_be_deleted):
            os.system(f'rm -r {dir_to_be_deleted}')

def main(dataset):
    delete_files(dataset)

def process_one_dataset(split_path, args):
    with open(split_path, 'r') as f:
        seq_name_list = f.readlines()
    seq_name_list = [seq_name.strip() for seq_name in seq_name_list]
    for seq_name in tqdm(seq_name_list):
        args.seq_name = seq_name
        dataset = get_dataset(args)
        main(dataset)

if __name__ == '__main__':
    args = get_args()
    
    # ScanNet
    split_path = 'splits/scannet.txt'
    args.dataset = 'scannet'
    process_one_dataset(split_path, args)

    # # ScanNet++
    # split_path = 'splits/scannetpp.txt'
    # args.dataset = 'scannetpp'
    # process_one_dataset(split_path, args)
    
    # # MatterPort3d
    # split_path = 'splits/matterport3d.txt'
    # args.dataset = 'matterport3d'
    # process_one_dataset(split_path, args)