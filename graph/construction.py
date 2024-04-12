import torch
import numpy as np
from tqdm import tqdm
from utils.mask_backprojection import frame_backprojection
from graph.node import Node

def mask_graph_construction(args, scene_points, frame_list, dataset):
    '''
        Construct the mask graph:
        1. Build the point in mask matrix. (To speed up the following computation of view consensus rate.)
        2. For each mask, compute the frames that it appears and the masks that contains it. Concurrently, we judge whether this mask is undersegmented.
        3. Build the nodes in the graph.
    '''
    if args.debug:
        print('start building point in mask matrix')
    boundary_points, point_in_mask_matrix, mask_point_clouds, point_frame_matrix, global_frame_mask_list = build_point_in_mask_matrix(args, scene_points, frame_list, dataset)
    visible_frames, contained_masks, undersegment_mask_ids = process_masks(frame_list, global_frame_mask_list, point_in_mask_matrix, boundary_points, mask_point_clouds, args)
    observer_num_thresholds = get_observer_num_thresholds(visible_frames)
    nodes = init_nodes(global_frame_mask_list, visible_frames, contained_masks, undersegment_mask_ids, mask_point_clouds)
    return nodes, observer_num_thresholds, mask_point_clouds, point_frame_matrix

def build_point_in_mask_matrix(args, scene_points, frame_list, dataset):
    '''
        To speed up the view consensus rate computation, we build a 'point in mask' matrix by a trade-off of space for time. This matrix is of size (scene_points_num, frame_num). For point i and frame j, if point i is in the k-th mask in frame j, then M[i,j] = k. Otherwise, M[i,j] = 0. (Note that mask id starts from 1).

        Returns:
            boundary_points: a set of points that are contained by multiple masks in a frame and thus are on the boundary of the masks. We will not consider these points in the following computation of view consensus rate.
            point_in_mask_matrix: the 'point in mask' matrix.
            mask_point_clouds: a dict where each key is the mask id in a frame, and the value is the point ids that are in this mask.
            point_frame_matrix: a matrix of size (scene_points_num, frame_num). For point i and frame j, if point i is visible in frame j, then M[i,j] = True. Otherwise, M[i,j] = False.
            global_frame_mask_list: a list of masks in the whole sequence. Each tuple contains the frame id and the mask id in this frame.
    '''
    
    scene_points_num = len(scene_points)
    frame_num = len(frame_list)

    scene_points = torch.tensor(scene_points).float().cuda()
    boundary_points = set()
    point_in_mask_matrix = np.zeros((scene_points_num, frame_num), dtype=np.uint16)
    point_frame_matrix = np.zeros((scene_points_num, frame_num), dtype=bool)
    global_frame_mask_list = []
    mask_point_clouds = {}
    
    iterator = tqdm(enumerate(frame_list), total=len(frame_list)) if args.debug else enumerate(frame_list)
    for frame_cnt, frame_id in iterator:
        mask_dict, frame_point_cloud_ids = frame_backprojection(dataset, scene_points, frame_id)
        if len(frame_point_cloud_ids) == 0:
            continue
        point_frame_matrix[frame_point_cloud_ids, frame_cnt] = True
        appeared_point_ids = set()
        frame_boundary_point_index = set()
        for mask_id, mask_point_cloud_ids in mask_dict.items():
            frame_boundary_point_index.update(mask_point_cloud_ids.intersection(appeared_point_ids))
            mask_point_clouds[f'{frame_id}_{mask_id}'] = mask_point_cloud_ids
            point_in_mask_matrix[list(mask_point_cloud_ids), frame_cnt] = mask_id
            appeared_point_ids.update(mask_point_cloud_ids)
            global_frame_mask_list.append((frame_id, mask_id))
        point_in_mask_matrix[list(frame_boundary_point_index), frame_cnt] = 0
        boundary_points.update(frame_boundary_point_index)
    
    return boundary_points, point_in_mask_matrix, mask_point_clouds, point_frame_matrix, global_frame_mask_list

def init_nodes(global_frame_mask_list, mask_project_on_all_frames, contained_masks, undersegment_mask_ids, mask_point_clouds):
    nodes = []
    for global_mask_id, (frame_id, mask_id) in enumerate(global_frame_mask_list):
        if global_mask_id in undersegment_mask_ids:
            continue
        mask_list = [(frame_id, mask_id)]
        frame = mask_project_on_all_frames[global_mask_id]
        frame_mask = contained_masks[global_mask_id]
        point_ids = mask_point_clouds[f'{frame_id}_{mask_id}']
        node_info = (0, len(nodes))
        node = Node(mask_list, frame, frame_mask, point_ids, node_info, None)
        nodes.append(node)
    return nodes

def get_observer_num_thresholds(visible_frames):
    '''
        Compute the observer number thresholds for each iteration. Range from 95% to 0%.
    '''
    observer_num_matrix = torch.matmul(visible_frames, visible_frames.transpose(0,1))
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

def process_one_mask(point_in_mask_matrix, boundary_points, mask_point_cloud, frame_list, global_frame_mask_list, args):
    '''
        For a mask, compute the frames that it is visible and the masks that contains it.
    '''
    visible_frame = torch.zeros(len(frame_list))
    contained_mask = torch.zeros(len(global_frame_mask_list))

    valid_mask_point_cloud = mask_point_cloud - boundary_points
    mask_point_cloud_info = point_in_mask_matrix[list(valid_mask_point_cloud), :]
    
    possibly_visible_frames = np.where(np.sum(mask_point_cloud_info, axis=0) > 0)[0]

    split_num = 0
    visible_num = 0
    
    for frame_id in possibly_visible_frames:
        mask_id_count = np.bincount(mask_point_cloud_info[:, frame_id])
        invisible_ratio = mask_id_count[0] / np.sum(mask_id_count) # 0 means that this point is invisible in this frame
        # If in a frame, most points in this mask are missing, then we think this mask is invisible in this frame.
        if 1 - invisible_ratio < args.mask_visible_threshold and (np.sum(mask_id_count) - mask_id_count[0]) < 500:
            continue
        visible_num += 1
        mask_id_count[0] = 0
        max_mask_id = np.argmax(mask_id_count)
        contained_ratio = mask_id_count[max_mask_id] / np.sum(mask_id_count)
        if contained_ratio > args.contained_threshold:
            visible_frame[frame_id] = 1
            frame_mask_idx = global_frame_mask_list.index((frame_list[frame_id], max_mask_id))
            contained_mask[frame_mask_idx] = 1
        else:
            split_num += 1 # This mask is splitted into two masks in this frame
    
    if visible_num == 0 or split_num / visible_num > args.undersegment_filter_threshold:
        return False, visible_frame, contained_mask
    else:
        return True, visible_frame, contained_mask

def process_masks(frame_list, global_frame_mask_list, point_in_mask_matrix, boundary_points, mask_point_clouds, args):
    '''
        For each mask, compute the frames that it is visible and the masks that contains it. 
        Meanwhile, we judge whether this mask is undersegmented.
    '''
    if args.debug:
        print('start processing masks')
    visible_frames = []
    contained_masks = []
    undersegment_mask_ids = []

    iterator = tqdm(global_frame_mask_list) if args.debug else global_frame_mask_list
    for frame_id, mask_id in iterator:
        valid, visible_frame, contained_mask = process_one_mask(point_in_mask_matrix, boundary_points, mask_point_clouds[f'{frame_id}_{mask_id}'], frame_list, global_frame_mask_list, args)
        visible_frames.append(visible_frame)
        contained_masks.append(contained_mask)
        if not valid:
            global_mask_id = global_frame_mask_list.index((frame_id, mask_id))
            undersegment_mask_ids.append(global_mask_id)

    visible_frames = torch.stack(visible_frames, dim=0).cuda() # (mask_num, frame_num)
    contained_masks = torch.stack(contained_masks, dim=0).cuda() # (mask_num, mask_num)

    # Undo the effect of undersegment observer masks to avoid merging two objects that are actually separated
    for global_mask_id in undersegment_mask_ids:
        frame_id, _ = global_frame_mask_list[global_mask_id]
        global_frame_id = frame_list.index(frame_id)
        mask_projected_idx = torch.where(contained_masks[:, global_mask_id])[0]
        contained_masks[:, global_mask_id] = False
        visible_frames[mask_projected_idx, global_frame_id] = False

    return visible_frames, contained_masks, undersegment_mask_ids