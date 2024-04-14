import os
from tqdm import tqdm
import time
from utils.config import get_args

CUDA_LIST = [1]

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

def get_seq_name_list(dataset):
    if dataset == 'scannet':
        file_path = 'splits/scannet.txt'
    elif dataset == 'scannetpp':
        file_path = 'splits/scannetpp.txt'
    elif dataset == 'matterport3d':
        file_path = 'splits/matterport3d.txt'
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
            command = f'CUDA_VISIBLE_DEVICES={cuda_id} {general_command % process_seq_name}'
            commands.append(command)
        execute_commands(commands, command_name, cuda_num)
    elif resource_type == 'cpu':
        commands = []
        for seq_name in seq_name_list:
            commands.append(f'{general_command} --seq_name {seq_name}')
        execute_commands(commands, command_name, cuda_num)

def get_label_text_feature(cuda_id):
    label_text_feature_path = 'data/text_features/matterport3d.npy'
    if os.path.exists(label_text_feature_path):
        return
    command = f'CUDA_VISIBLE_DEVICES={cuda_id} python -m semantics.extract_label_featrues'
    os.system(command)

def main(args):
    dataset = args.dataset
    config = args.config
    cropformer_path = args.cropformer_path

    if dataset == 'scannet':
        root = 'data/scannet/processed'
        image_path_pattern = 'color/*0.jpg' # stride = 10
        gt = 'data/scannet/gt'
    elif dataset == 'scannetpp':
        root = 'data/scannetpp/data'
        image_path_pattern = 'iphone/rgb/*0.jpg'
        gt = 'data/scannetpp/gt'
    elif dataset == 'matterport3d':
        root = 'data/matterport3d/scans'
        image_path_pattern = '*/undistorted_color_images/*.jpg' # stride = 1
        gt = 'data/matterport3d/gt'

    t0 = time.time()
    seq_name_list = get_seq_name_list(dataset)
    print('There are %d scenes' % len(seq_name_list))
    
    # Step 1: use Cropformer to get 2D instance masks for all sequences.
    parallel_compute(f'python third_party/detectron2/projects/CropFormer/demo_cropformer/mask_predict.py --config-file third_party/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml --root {root} --image_path_pattern {image_path_pattern} --dataset {args.dataset} --seq_name_list %s --opts MODEL.WEIGHTS {cropformer_path}', 'predict mask', 'cuda', CUDA_LIST, seq_name_list)

    # # Step 2: Mask clustering using our proposed method.
    parallel_compute(f'python main.py --config {config} --seq_name_list %s', 'mask clustering', 'cuda', CUDA_LIST, seq_name_list)
    
    # Step 3: Evaluate the class-agnostic results.
    os.system(f'python -m evaluation.evaluate --pred_path data/prediction/{config}_class_agnostic --gt_path {gt} --dataset {dataset} --no_class')

    # Step 4: Get the open-vocabulary semantic features for each 2D masks.
    parallel_compute(f'python -m semantics.get_open-voc_features --config {config}  --seq_name_list %s', 'get open-vocabulary semantic features using CLIP', 'cuda', CUDA_LIST, seq_name_list)

    # Step 5: Get the text CLIP features for each label.
    get_label_text_feature(CUDA_LIST[0])
    
    # Step 6: Get labels for each 3D instances.
    parallel_compute(f'python -m semantics.open-voc_query --config {config}', 'get text labels', 'cpu', CUDA_LIST, seq_name_list)
    
    # Step 7: Evaluate the class-aware results.
    os.system(f'python -m evaluation.evaluate --pred_path data/prediction/{config} --gt_path {gt} --dataset {dataset}')

    print('total time', (time.time() - t0)//60, 'min')
    print('Average time', (time.time() - t0) / len(seq_name_list), 'sec')

if __name__ == '__main__':
    args = get_args()
    main(args)