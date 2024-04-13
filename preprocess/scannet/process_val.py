import os
from tqdm import tqdm
import shutil

raw_data_dir = '../../data/scannet/raw/scans'
target_data_dir = '../../data/scannet/processed'
split_file_path = '../../splits/scannet.txt'

def process_one_seq(seq_name):
    target_seq_dir = os.path.join(target_data_dir, seq_name)
    os.makedirs(target_data_dir, exist_ok=True)

    command = f'python reader.py --filename {raw_data_dir}/{seq_name}/{seq_name}.sens --output_path {target_seq_dir} --export_color_images --export_depth_images --export_poses --export_intrinsics'
    os.system(command)

    # copy the point cloud file here
    shutil.copyfile(f'{raw_data_dir}/{seq_name}/{seq_name}_vh_clean_2.ply', f'{target_seq_dir}/{seq_name}_vh_clean_2.ply')

with open(split_file_path, 'r') as f:
    seq_name_list = f.readlines()
seq_name_list = [seq_name.strip() for seq_name in seq_name_list]

for seq_name in tqdm(seq_name_list):
    process_one_seq(seq_name)