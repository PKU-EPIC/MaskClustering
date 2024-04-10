import os
from tqdm import tqdm
import time

def execute_commands(commands_list, command_type, process_num):
    print('====> Start', command_type)
    from multiprocessing import Pool
    pool = Pool(process_num)
    for _ in tqdm(pool.imap_unordered(os.system, commands_list), total=len(commands_list)):
        pass
    pool.close()
    pool.join()
    pool.terminate()
    print('====> Finish', command_type)

def get_seq_name_list(dataset_type):
    file_path = '/home/miyan/3DSAM/data/scannet/splits/scannetv2_val.txt'
    with open(file_path, 'r') as f:
        seq_name_list = f.readlines()
    seq_name_list = [seq_name.strip() for seq_name in seq_name_list]
    return seq_name_list

def parallel_compute(general_command, command_name, resource_type, cuda_list, seq_name_list):
    cuda_num = len(cuda_list)
    
    if resource_type == 'cuda':
        commands = []
        for i, cuda_id in enumerate(cuda_list):
            process_seq_name = seq_name_list[i::cuda_num]
            if len(process_seq_name) == 0:
                continue
            process_seq_name = '+'.join(process_seq_name)
            command = f'CUDA_VISIBLE_DEVICES={cuda_id} {general_command} --seq_name_list {process_seq_name}'
            commands.append(command)
        execute_commands(commands, command_name, cuda_num)
    elif resource_type == 'cpu':
        commands = []
        for seq_name in seq_name_list:
            commands.append(f'{general_command} --seq_name {seq_name}')
        execute_commands(commands, command_name, cuda_num)

def main(config_type):
    dataset_type = 'scannet'
    root = '/home/miyan/3DSAM/data/scannet/processed'
    cuda_list = [0, 1]
    step = 10
    os.makedirs(f'/home/miyan/3DSAM/data/scannet/instance_segmentation/pred_10_{config_type}/', exist_ok=True)

    t0 = time.time()
    seq_name_list = get_seq_name_list(dataset_type)
    print('There are %d scenes' % len(seq_name_list))
    
    parallel_compute(f'python detectron2/projects/CropFormer/demo_cropformer/mask_predict.py --config-file detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml --root {root} --image_path_pattern \'color_640/*0.jpg\'', 'predict mask', 'cuda', cuda_list, seq_name_list)

    parallel_compute(f'python mask_clustering.py --dataset_type {dataset_type} --step {step} --config_type {config_type}', 'mask clustering', 'cuda', cuda_list, seq_name_list)

    parallel_compute(f'python -m semantics.get_open-voc_features --dataset_type {dataset_type} --step {step} --config_type {config_type}', 'get open-vocabulary semantic features using CLIP', 'cuda', cuda_list, seq_name_list)

    parallel_compute(f'python -m semantics.open-voc_query --dataset_type {dataset_type} --step {step} --config_type {config_type}', 'get text labels', 'cpu', cuda_list, seq_name_list)
    
    os.system(f'python -m evaluation.evaluate --pred_path data/scannet/instance_segmentation/{config_type} --gt_path /home/miyan/3Dmapping/data/scannet/instance_segmentation/gt/gt_file/200 --dataset scannet')
    os.system(f'python -m evaluation.evaluate --pred_path data/scannet/instance_segmentation/{config_type} --gt_path /home/miyan/3Dmapping/data/scannet/instance_segmentation/gt/gt_file/200 --dataset scannet --no_class')

    print('total time', (time.time() - t0)//60)
    print('Total scenes', len(seq_name_list))
    print('Average time', (time.time() - t0) / len(seq_name_list))

if __name__ == '__main__':
    for config_type in ['scannet', 'scannet_filter_0.5']:
        main(config_type)