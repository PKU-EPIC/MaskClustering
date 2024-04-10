import numpy as np
import open3d as o3d
import os
import networkx as nx
import numpy as np
import torch
from utils.config import get_dataset, get_args
from tqdm import tqdm
from utils.mask_backprojection import frame_backprojection
import sys

class Segment:
    def __init__(self, mask_list, frame, frame_mask, complete_vertex_index, node_info, son_node_info):
        self.mask_list = mask_list
        self.frame = frame # one-hot vector
        self.frame_mask = frame_mask # one-hot vector
        self.complete_vertex_index = complete_vertex_index
        self.node_info = node_info
        self.son_node_info = son_node_info
    
    @ staticmethod
    def create_segment_from_list(segment_list, node_info):
        mask_list = []
        frame_one_hot = torch.zeros(len(segment_list[0].frame), dtype=bool).cuda()
        frame_mask_one_hot = torch.zeros(len(segment_list[0].frame_mask), dtype=bool).cuda()
        complete_vertex_index = set()
        son_node_info = set()
        for segment in segment_list:
            mask_list += segment.mask_list
            frame_one_hot = frame_one_hot | (segment.frame).bool()
            frame_mask_one_hot = frame_mask_one_hot | (segment.frame_mask).bool()
            complete_vertex_index = complete_vertex_index.union(segment.complete_vertex_index)
            son_node_info.add(segment.node_info)
        return Segment(mask_list, frame_one_hot.float(), frame_mask_one_hot.float(), complete_vertex_index, node_info, son_node_info)
    
    def get_complete_pcld(self, coarse_points, coarse_colors):
        complete_index_list = list(self.complete_vertex_index)
        points = coarse_points[complete_index_list]
        colors = coarse_colors[complete_index_list]
        pcld = o3d.geometry.PointCloud()
        pcld.points = o3d.utility.Vector3dVector(points)
        pcld.colors = o3d.utility.Vector3dVector(colors)
        return pcld, complete_index_list

def build_raw_matrix(scene_points, coarse_points_num, frame_list, dataset):
    '''
        Iterate over all masks to build the raw point mask matrix where M[i,j] stores the mask id for point i in frame j
        If point i is not in any mask in frame j, M[i,j] = 0. (Note that mask id starts from 1).

        We also build a coarse_point_frame_matrix where P[i,j] stores the number of masks that point i appears in frame j. This matrix is used to decide whether a point is a boundary point.

        At each frame, we maintain a frame_boundary_point_index to record all boundary points in this frame and set the mask id of these points to 0.
    '''
    
    scene_points = torch.tensor(scene_points).float().cuda()
    total_frame = len(frame_list)
    boundary_point_index = set()
    point_mask_matrix = np.zeros((coarse_points_num, total_frame), dtype=np.uint16)
    coarse_point_frame_matrix = np.zeros((coarse_points_num, total_frame), dtype=bool)
    frame_mask_list = []

    mask_complete_vertex_index = {}

    if args.debug:
        iterator = tqdm(enumerate(frame_list), total=len(frame_list))
    else:
        iterator = enumerate(frame_list)
    
    for frame_cnt, frame_id in iterator:
        mask_dict, frame_point_cloud_ids = frame_backprojection(dataset, scene_points, frame_id)
        if len(frame_point_cloud_ids) == 0:
            continue
        coarse_point_frame_matrix[frame_point_cloud_ids, frame_cnt] = True
        appeared_vertex_index = set()
        frame_boundary_point_index = set()
        for mask_id, mask_point_cloud_ids in mask_dict.items():
            frame_boundary_point_index.update(mask_point_cloud_ids.intersection(appeared_vertex_index))
            mask_complete_vertex_index[f'{frame_id}_{mask_id}'] = mask_point_cloud_ids
            point_mask_matrix[list(mask_point_cloud_ids), frame_cnt] = mask_id
            appeared_vertex_index.update(mask_point_cloud_ids)
            frame_mask_list.append((frame_id, mask_id))
        point_mask_matrix[list(frame_boundary_point_index), frame_cnt] = 0
        boundary_point_index.update(frame_boundary_point_index)
    
    return boundary_point_index, point_mask_matrix, mask_complete_vertex_index, coarse_point_frame_matrix, frame_mask_list

def project_one_mask_on_all_frames(point_mask_matrix, invalid_point_set, frame_id, mask_id, mask_complete_vertex_index, frame_list, frame_mask_list,  args, debug=False):
    '''
        This function judges whether a mask is undersegment and returns its corresponding mask id in other frames
    '''
    frame_one_hot = torch.zeros(len(frame_list))
    frame_mask_one_hot = torch.zeros(len(frame_mask_list))
    
    mask_vertex_index = mask_complete_vertex_index[f'{frame_id}_{mask_id}']
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

        if debug:
            print(frame_list[frame_cnt], ratio, np.where(mask_id_count>0)[0], np.sum(mask_id_count) - mask_id_count[max_mask_id])
    
    if appear_num == 0:
        return False, frame_one_hot, frame_mask_one_hot
    
    invalid_ratio = split_num / appear_num

    if invalid_ratio > args.undersegment_mask_ratio:
        return False, frame_one_hot, frame_mask_one_hot
    else:
        return True, frame_one_hot, frame_mask_one_hot

def remove_undersegment_mask(frame_list, frame_mask_list, point_mask_matrix, invalid_point_set, mask_complete_vertex_index, args):
    '''
        mask_project_on_all_frames: a dict where each key is the global mask id, i.e., '{frame_id}_{mask_id}'. Each value contains two elements, frame and frame_mask, the former records what frames this key appears, the latter records what the corresponding mask id in these frames.
            example: '100_10': ((0,10,500), ((0,1), (10,2), (500,9)))
    '''
    mask_project_on_all_frames = []
    mask_project_on_all_frame_masks = []

    # find all undersegment masks and build mask_project_on_all_frames
    undersegment_mask_ids = []
    if args.debug:
        iterator = tqdm(frame_mask_list)
    else:
        iterator = frame_mask_list

    for frame_id, mask_id in iterator:
        valid, frame_one_hot, frame_mask_one_hot = project_one_mask_on_all_frames(point_mask_matrix, invalid_point_set, frame_id, mask_id, mask_complete_vertex_index, frame_list, frame_mask_list, args)
        mask_project_on_all_frames.append(frame_one_hot)
        mask_project_on_all_frame_masks.append(frame_mask_one_hot)
        if not valid:
            global_mask_id = frame_mask_list.index((frame_id, mask_id))
            undersegment_mask_ids.append(global_mask_id)

    mask_project_on_all_frames = torch.stack(mask_project_on_all_frames, dim=0).cuda()
    mask_project_on_all_frame_masks = torch.stack(mask_project_on_all_frame_masks, dim=0).cuda()

    # remove undersegment masks
    for global_mask_id in undersegment_mask_ids:
        frame_id, _ = frame_mask_list[global_mask_id]
        global_frame_id = frame_list.index(frame_id)
        mask_projected_idx = torch.where(mask_project_on_all_frame_masks[:, global_mask_id])[0]
        mask_project_on_all_frame_masks[:, global_mask_id] = False
        mask_project_on_all_frames[mask_projected_idx, global_frame_id] = False

    return mask_project_on_all_frames, mask_project_on_all_frame_masks, undersegment_mask_ids

def get_third_view_num_list(mask_project_on_all_frames):
    third_view_num_matrix = torch.matmul(mask_project_on_all_frames, mask_project_on_all_frames.transpose(0,1))
    third_view_num_list = third_view_num_matrix.flatten()
    third_view_num_list = third_view_num_list[third_view_num_list > 0]
    return third_view_num_list.cpu().numpy()

def prepare_raw_segment_list(frame_mask_list, mask_project_on_all_frames, mask_project_on_all_frame_masks, undersegment_mask_ids, mask_complete_vertex_index):
    raw_to_be_processed = []
    for global_mask_id, (frame_id, mask_id) in enumerate(frame_mask_list):
        if global_mask_id in undersegment_mask_ids:
            continue
        mask_list = [(frame_id, mask_id)]
        frame = mask_project_on_all_frames[global_mask_id]
        frame_mask = mask_project_on_all_frame_masks[global_mask_id]
        complete_vertex_index = mask_complete_vertex_index[f'{frame_id}_{mask_id}']
        node_info = (0, len(raw_to_be_processed))
        segment = Segment(mask_list, frame, frame_mask, complete_vertex_index, node_info, None)
        raw_to_be_processed.append(segment)
    return raw_to_be_processed

def build_graph(segment_list, third_view_num, SEGMENT_CONNECT_RATIO):
    segment_frame_matrix = torch.stack([segment.frame for segment in segment_list], dim=0)
    segment_frame_mask_matrix = torch.stack([segment.frame_mask for segment in segment_list], dim=0)

    same_frame_matrix = torch.matmul(segment_frame_matrix, segment_frame_matrix.transpose(0,1))
    same_frame_mask_matrix = torch.matmul(segment_frame_mask_matrix, segment_frame_mask_matrix.transpose(0,1))

    disconnect_mask = torch.eye(len(segment_list), dtype=bool).cuda()
    disconnect_mask = disconnect_mask | (same_frame_matrix < third_view_num)

    concensus_rate = same_frame_mask_matrix / (same_frame_matrix + 1e-7)
    A = concensus_rate >= SEGMENT_CONNECT_RATIO
    A = A & ~disconnect_mask
    A = A.cpu().numpy()

    G = nx.from_numpy_array(A)
    return G

def merge_into_new_segment_list(level, old_segment_list, graph):
    segment_list = []
    for component in nx.connected_components(graph):
        node_info = (level, len(segment_list))
        segment_list.append(Segment.create_segment_from_list([old_segment_list[node] for node in component], node_info))
    return segment_list

def mask_association(raw_to_be_processed, third_view_num_list, SEGMENT_CONNECT_RATIO, debug):
    segment_list = raw_to_be_processed
    for i, percentile in enumerate(range(95, -5, -5)):
        third_view_num = np.percentile(third_view_num_list, percentile)
        if third_view_num <= 1:
            if percentile < 50:
                break
            else:
                third_view_num = 1
        if debug:
            print('third_view_num', third_view_num, 'number of nodes', len(segment_list))

        graph = build_graph(segment_list, third_view_num, SEGMENT_CONNECT_RATIO)
        segment_list = merge_into_new_segment_list(i+1, segment_list, graph)
    return segment_list

def judge_bbox_overlay(bbox_1, bbox_2):
    for i in range(3):
        if bbox_1[0][i] > bbox_2[1][i] or bbox_2[0][i] > bbox_1[1][i]:
            return False
    return True

def merge_overlapping_objects(total_pcld_list, total_pcld_index_list, total_pcld_bbox_list, total_object_mask_list, overlapping_ratio):
    total_object_num = len(total_pcld_list)
    invalid_object = np.zeros(total_object_num, dtype=bool)

    for i in range(total_object_num):
        if invalid_object[i]:
            continue
        pcld_index_i = set(total_pcld_index_list[i])
        bbox_i = total_pcld_bbox_list[i]
        for j in range(i+1, total_object_num):
            if invalid_object[j]:
                continue
            pcld_index_j = set(total_pcld_index_list[j])
            bbox_j = total_pcld_bbox_list[j]
            if judge_bbox_overlay(bbox_i, bbox_j):
                intersect = len(pcld_index_i.intersection(pcld_index_j))
                if intersect / len(pcld_index_i) > overlapping_ratio:
                    invalid_object[i] = True
                elif intersect / len(pcld_index_j) > overlapping_ratio:
                    invalid_object[j] = True

    valid_pcld_list = []
    valid_pcld_index_list = []
    valid_pcld_mask_list = []
    for i in range(total_object_num):
        if not invalid_object[i]:
            valid_pcld_list.append(total_pcld_list[i])
            valid_pcld_index_list.append(total_pcld_index_list[i])
            valid_pcld_mask_list.append(total_object_mask_list[i])
    return valid_pcld_list, valid_pcld_index_list, valid_pcld_mask_list

def point_filter_and_coverage_computing(coarse_point_frame_matrix, segment, object_pcld_list, object_pcld_coarse_index_list, mask_complete_vertex_index, frame_list, args):
    segment_global_frame_id_list = torch.where(segment.frame)[0].cpu().numpy()
    segment_frame_id_list = np.array(frame_list)[segment_global_frame_id_list]
    mask_list = segment.mask_list

    point_appear_matrix_list = []
    point_within_segment_list = []
    for object_pcld_coarse_index in object_pcld_coarse_index_list:
        point_appear_matrix = coarse_point_frame_matrix[object_pcld_coarse_index, ]
        point_appear_matrix = point_appear_matrix[:, segment_global_frame_id_list]
        point_appear_matrix_list.append(point_appear_matrix)
        point_within_segment = np.zeros_like(point_appear_matrix, dtype=bool)
        point_within_segment_list.append(point_within_segment)

    object_mask_list = []
    for _ in range(len(object_pcld_list)):
        object_mask_list.append([])

    # compute mask coverage
    for frame_id, mask_id in (mask_list):
        frame_id_in_list = np.where(segment_frame_id_list == frame_id)[0][0]

        mask_vertex_idx = list(mask_complete_vertex_index[f'{frame_id}_{mask_id}'])
        mask_coarse_vertex_idx = mask_vertex_idx
        max_match_object_idx = -1
        max_intersect = 0
        coverage_list = []
        for i, object_pcld_coarse_index in enumerate(object_pcld_coarse_index_list):
            indices = np.where(np.isin(object_pcld_coarse_index, mask_coarse_vertex_idx))[0]
            point_within_segment_list[i][indices, frame_id_in_list] = True
            if len(indices) > max_intersect:
                max_intersect = len(indices)
                max_match_object_idx = i
            coverage = len(indices) / len(object_pcld_coarse_index)
            coverage_list.append(coverage)
        if max_intersect == 0:
            continue
        object_mask_list[max_match_object_idx] += [(frame_id, mask_id, coverage_list[max_match_object_idx])]

    # filter points
    filtered_object_pcld_list = []
    filtered_object_pcld_coarse_index_list = []
    filtered_object_mask_list = []
    filtered_object_bbox_list = []
    for i, (point_appear_matrix, point_within_segment) in enumerate(zip(point_appear_matrix_list, point_within_segment_list)):
        detection_ratio = np.sum(point_within_segment, axis=1) / np.sum(point_appear_matrix, axis=1) + 1e-6
        valid_point_index = np.where(detection_ratio > args.point_filter_threshold)[0]
        if len(valid_point_index) == 0:
            continue
        filtered_object_pcld_list.append(object_pcld_list[i].select_by_index(valid_point_index))
        filtered_object_pcld_coarse_index_list.append(object_pcld_coarse_index_list[i][valid_point_index])
        filtered_object_bbox_list.append([np.amin(object_pcld_list[i].points, axis=0), np.amax(object_pcld_list[i].points, axis=0)])
        filtered_object_mask_list.append(object_mask_list[i])
    return filtered_object_pcld_list, filtered_object_pcld_coarse_index_list, filtered_object_bbox_list, filtered_object_mask_list

def dbscan_process(pcld, complete_index_list, DBSCAN_THRESHOLD=0.2):
    '''
        dbscan splitting
    '''
    
    labels = np.array(pcld.cluster_dbscan(eps=DBSCAN_THRESHOLD, min_points=4)) + 1 # -1 for noise
    count = np.bincount(labels)

    # split disconnected point cloud into different objects
    pcld_list, pcld_index_list = [], []
    pcld_idx_list = np.array(complete_index_list)
    for i in range(len(count)):
        remain_index = np.where(labels == i)[0]
        if len(remain_index) == 0:
            continue
        object_pcld = pcld.select_by_index(remain_index)
        pcld_index = pcld_idx_list[remain_index]
        pcld_list.append(object_pcld)
        pcld_index_list.append(pcld_index)
    return pcld_list, pcld_index_list

def find_represent_mask(mask_info_list):
    mask_info_list.sort(key=lambda x: x[2], reverse=True)
    return mask_info_list[:5]

def export_objects(dataset, segment_list, mask_complete_vertex_index, coarse_points, coarse_colors, coarse_point_frame_matrix, frame_list, args):
    total_pcld_list = []
    total_pcld_index_list = []
    total_pcld_bbox_list = []
    total_object_mask_list = []
    
    for i, segment in enumerate(segment_list):
        if len(segment.mask_list) < 2:
            continue
        
        pcld, complete_index_list = segment.get_complete_pcld(coarse_points, coarse_colors)
        object_pcld_list, object_pcld_coarse_index_list = dbscan_process(pcld, complete_index_list, args.dbscan_threshold)

        object_pcld_list, object_pcld_coarse_index_list, object_bbox_list, object_mask_list = point_filter_and_coverage_computing(coarse_point_frame_matrix, segment, object_pcld_list, object_pcld_coarse_index_list, mask_complete_vertex_index, frame_list, args)

        total_pcld_list.extend(object_pcld_list)
        total_pcld_index_list.extend(object_pcld_coarse_index_list)
        total_pcld_bbox_list.extend(object_bbox_list)
        total_object_mask_list.extend(object_mask_list)

    total_pcld_list, total_pcld_index_list, total_pcld_mask_list = merge_overlapping_objects(total_pcld_list, total_pcld_index_list, total_pcld_bbox_list, total_object_mask_list, args.overlapping_ratio)

    object_dict = {}
    for i, (pcld, vertex_index, pcld_mask_list) in enumerate(zip(total_pcld_list, total_pcld_index_list, total_pcld_mask_list)):
        object_dict[i] = {
            'vertex_index': vertex_index,
            'mask_list': pcld_mask_list,
            'repre_mask_list': find_represent_mask(pcld_mask_list),
        }
    np.save(os.path.join(dataset.object_dict_dir, args.config, f'object_dict.npy'), object_dict, allow_pickle=True)
    return

def main(args):
    dataset = get_dataset(args)
    # if os.path.exists(os.path.join(dataset.object_dict_dir, args.config, f'object_dict.npy')):
    #     return

    coarse_pcld = o3d.io.read_point_cloud(dataset.mesh_path)
    coarse_points = np.asarray(coarse_pcld.points)
    coarse_colors = np.zeros_like(coarse_points)

    os.makedirs(os.path.join(dataset.object_dict_dir, args.config), exist_ok=True)

    frame_list = list(dataset.get_frame_list(0, -1, args.step))
    if args.debug:
        print('start building point_mask_matrix')
    boundary_point_index, point_mask_matrix, mask_complete_vertex_index, coarse_point_frame_matrix, frame_mask_list = build_raw_matrix(coarse_points, len(coarse_points), frame_list, dataset)
    invalid_point_set = boundary_point_index
    if args.debug:
        print('start removing undersegment mask')
    
    with torch.no_grad():
        mask_project_on_all_frames, mask_project_on_all_frame_masks, undersegment_mask_ids = remove_undersegment_mask(frame_list, frame_mask_list, point_mask_matrix, invalid_point_set, mask_complete_vertex_index, args)
        third_view_num_list = get_third_view_num_list(mask_project_on_all_frames)
        if args.debug:
            print('start prepare raw segment list')
        raw_to_be_processed = prepare_raw_segment_list(frame_mask_list, mask_project_on_all_frames, mask_project_on_all_frame_masks, undersegment_mask_ids, mask_complete_vertex_index)
        if args.debug:
            print('start associate mask')
        object_list = mask_association(raw_to_be_processed, third_view_num_list, args.segment_connect_ratio, args.debug)

        if args.debug:
            print('start exporting')
        export_objects(dataset, object_list, mask_complete_vertex_index, coarse_points, coarse_colors, coarse_point_frame_matrix, frame_list, args)

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
