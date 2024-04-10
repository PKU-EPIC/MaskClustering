import torch
import numpy as np
from tqdm import tqdm
from utils.mask_backprojection import frame_backprojection
from graph.node import Node

def mask_graph_construction(args, scene_points, frame_list, dataset):
    if args.debug:
        print('start building point in mask matrix')
    boundary_points, point_mask_matrix, mask_point_clouds, coarse_point_frame_matrix, frame_mask_list = build_point_in_mask_matrix(args, scene_points, frame_list, dataset)
    
    mask_project_on_all_frames, mask_project_on_all_frame_masks, undersegment_mask_ids = process_mask(frame_list, frame_mask_list, point_mask_matrix, boundary_points, mask_point_clouds, args)
    observer_num_thresholds = get_observer_num_thresholds(mask_project_on_all_frames)

    nodes = get_nodes(frame_mask_list, mask_project_on_all_frames, mask_project_on_all_frame_masks, undersegment_mask_ids, mask_point_clouds)

    return nodes, observer_num_thresholds, mask_point_clouds, coarse_point_frame_matrix

def build_point_in_mask_matrix(args, scene_points, frame_list, dataset):
    '''
        To speed up the view consensus rate computation, we build a 'point in mask' matrix by a trade-off of space for time. This matrix is of size (scene_points_num, total_frame). For point i and frame j, if point i is in the k-th mask in frame j, then M[i,j] = k. Otherwise, M[i,j] = 0. (Note that mask id starts from 1).

        We also build a coarse_point_frame_matrix where P[i,j] stores the number of masks that point i appears in frame j. This matrix is used to decide whether a point is a boundary point.

        At each frame, we maintain a frame_boundary_point_index to record all boundary points in this frame and set the mask id of these points to 0.
    '''
    
    scene_points_num = len(scene_points)
    scene_points = torch.tensor(scene_points).float().cuda()
    total_frame = len(frame_list)
    boundary_points = set()
    point_mask_matrix = np.zeros((scene_points_num, total_frame), dtype=np.uint16)
    coarse_point_frame_matrix = np.zeros((scene_points_num, total_frame), dtype=bool)
    frame_mask_list = []

    mask_point_clouds = {}
    
    iterator = tqdm(enumerate(frame_list), total=len(frame_list)) if args.debug else enumerate(frame_list)
    
    for frame_cnt, frame_id in iterator:
        mask_dict, frame_point_cloud_ids = frame_backprojection(dataset, scene_points, frame_id)
        if len(frame_point_cloud_ids) == 0:
            continue
        coarse_point_frame_matrix[frame_point_cloud_ids, frame_cnt] = True
        appeared_vertex_index = set()
        frame_boundary_point_index = set()
        for mask_id, mask_point_cloud_ids in mask_dict.items():
            frame_boundary_point_index.update(mask_point_cloud_ids.intersection(appeared_vertex_index))
            mask_point_clouds[f'{frame_id}_{mask_id}'] = mask_point_cloud_ids
            point_mask_matrix[list(mask_point_cloud_ids), frame_cnt] = mask_id
            appeared_vertex_index.update(mask_point_cloud_ids)
            frame_mask_list.append((frame_id, mask_id))
        point_mask_matrix[list(frame_boundary_point_index), frame_cnt] = 0
        boundary_points.update(frame_boundary_point_index)
    
    return boundary_points, point_mask_matrix, mask_point_clouds, coarse_point_frame_matrix, frame_mask_list

def get_nodes(frame_mask_list, mask_project_on_all_frames, mask_project_on_all_frame_masks, undersegment_mask_ids, mask_point_clouds):
    raw_to_be_processed = []
    for global_mask_id, (frame_id, mask_id) in enumerate(frame_mask_list):
        if global_mask_id in undersegment_mask_ids:
            continue
        mask_list = [(frame_id, mask_id)]
        frame = mask_project_on_all_frames[global_mask_id]
        frame_mask = mask_project_on_all_frame_masks[global_mask_id]
        complete_vertex_index = mask_point_clouds[f'{frame_id}_{mask_id}']
        node_info = (0, len(raw_to_be_processed))
        segment = Node(mask_list, frame, frame_mask, complete_vertex_index, node_info, None)
        raw_to_be_processed.append(segment)
    return raw_to_be_processed

def get_observer_num_thresholds(mask_project_on_all_frames):
    observer_num_matrix = torch.matmul(mask_project_on_all_frames, mask_project_on_all_frames.transpose(0,1))
    observer_num_list = observer_num_matrix.flatten()
    observer_num_list = observer_num_list[observer_num_list > 0].cpu().numpy()
    observer_num_thresholds = []
    for percentile in range(95, -5, -5):
        observer_num = np.percentile(observer_num_list, percentile)
        if observer_num <= 1:
            if percentile < 50:
                break
            else:
                observer_num = 1
        observer_num_thresholds.append(observer_num)
    return observer_num_thresholds

def project_one_mask_on_all_frames(point_mask_matrix, invalid_point_set, frame_id, mask_id, mask_point_clouds, frame_list, frame_mask_list, args):
    '''
        This function judges whether a mask is undersegment and returns its corresponding mask id in other frames
    '''
    frame_one_hot = torch.zeros(len(frame_list))
    frame_mask_one_hot = torch.zeros(len(frame_mask_list))
    
    mask_vertex_index = mask_point_clouds[f'{frame_id}_{mask_id}']
    valid_mask_vertex_index = mask_vertex_index - invalid_point_set
    project_on_all_frames = point_mask_matrix[list(valid_mask_vertex_index), :]
    
    possible_frames_list = np.where(np.sum(project_on_all_frames, axis=0) > 0)[0]

    split_num = 0
    appear_num = 0
    
    for frame_cnt in possible_frames_list:
        mask_id_count = np.bincount(project_on_all_frames[:, frame_cnt])
        disappear_ratio = mask_id_count[0] / np.sum(mask_id_count)
        # If in a frame, most of the points in the mask are missing, then we simply ignore this frame
        if disappear_ratio > args.mask_disappear_ratio and (np.sum(mask_id_count) - mask_id_count[0]) < args.mask_disappear_num:
            continue
        appear_num += 1
        mask_id_count[0] = 0
        max_mask_id = np.argmax(mask_id_count)
        ratio = mask_id_count[max_mask_id] / np.sum(mask_id_count)
        if ratio > args.valid_mask_ratio:
            frame_one_hot[frame_cnt] = 1
            frame_mask_idx = frame_mask_list.index((frame_list[frame_cnt], max_mask_id))
            frame_mask_one_hot[frame_mask_idx] = 1
        else:
            split_num += 1

    if appear_num == 0:
        return False, frame_one_hot, frame_mask_one_hot
    
    invalid_ratio = split_num / appear_num

    if invalid_ratio > args.undersegment_mask_ratio:
        return False, frame_one_hot, frame_mask_one_hot
    else:
        return True, frame_one_hot, frame_mask_one_hot

def process_mask(frame_list, frame_mask_list, point_mask_matrix, invalid_point_set, mask_point_clouds, args):
    '''
        mask_project_on_all_frames: a dict where each key is the global mask id, i.e., '{frame_id}_{mask_id}'. Each value contains two elements, frame and frame_mask, the former records what frames this key appears, the latter records what the corresponding mask id in these frames.
            example: '100_10': ((0,10,500), ((0,1), (10,2), (500,9)))
    '''
    visible_frame_list = []
    mask_project_on_all_frame_masks = []

    # find all undersegment masks and build mask_project_on_all_frames
    undersegment_mask_ids = []
    if args.debug:
        iterator = tqdm(frame_mask_list)
    else:
        iterator = frame_mask_list

    for frame_id, mask_id in iterator:
        valid, frame_one_hot, frame_mask_one_hot = project_one_mask_on_all_frames(point_mask_matrix, invalid_point_set, frame_id, mask_id, mask_point_clouds, frame_list, frame_mask_list, args)
        visible_frame_list.append(frame_one_hot)
        mask_project_on_all_frame_masks.append(frame_mask_one_hot)
        if not valid:
            global_mask_id = frame_mask_list.index((frame_id, mask_id))
            undersegment_mask_ids.append(global_mask_id)

    visible_frame_list = torch.stack(visible_frame_list, dim=0).cuda()
    mask_project_on_all_frame_masks = torch.stack(mask_project_on_all_frame_masks, dim=0).cuda()

    # remove undersegment masks
    for global_mask_id in undersegment_mask_ids:
        frame_id, _ = frame_mask_list[global_mask_id]
        global_frame_id = frame_list.index(frame_id)
        mask_projected_idx = torch.where(mask_project_on_all_frame_masks[:, global_mask_id])[0]
        mask_project_on_all_frame_masks[:, global_mask_id] = False
        visible_frame_list[mask_projected_idx, global_frame_id] = False

    return visible_frame_list, mask_project_on_all_frame_masks, undersegment_mask_ids