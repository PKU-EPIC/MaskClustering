import torch
from utils.config import get_dataset, get_args
from utils.post_process import export_objects
from graph.construction import mask_graph_construction
from graph.iterative_clustering import iterative_clustering
import sys
import os

def main(args):
    dataset = get_dataset(args)
    scene_points = dataset.get_scene_points()
    frame_list = dataset.get_frame_list(args.step)
    # if os.path.exists(os.path.join(dataset.object_dict_dir, args.config, f'object_dict.npy')):
    #     return

    with torch.no_grad():
        nodes, observer_num_thresholds, mask_point_clouds, point_frame_matrix = mask_graph_construction(args, scene_points, frame_list, dataset)

        object_list = iterative_clustering(nodes, observer_num_thresholds, args.segment_connect_ratio, args.debug)

        export_objects(dataset, object_list, mask_point_clouds, scene_points, point_frame_matrix, frame_list, args)

if __name__ == '__main__':
    import time
    args = get_args()
    seq_name_list = args.seq_name_list.split('+')

    t0 = time.time()
    for i, seq_name in enumerate(seq_name_list):
        args.seq_name = seq_name
        main(args)
        t1 = time.time()
        time_left = (t1 - t0) / (i+1) * (len(seq_name_list) - i - 1) // 60
        sys.stdout.write(f'\rFinish {i+1}/{len(seq_name_list)} sequences.' + f' Time left: {time_left} min')
        sys.stdout.flush()