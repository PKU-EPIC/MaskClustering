import numpy as np
import os
import torch
from utils.geometry import denoise, judge_bbox_overlay


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


def point_filter_and_coverage_computing(coarse_point_frame_matrix, segment, object_pcld_list, object_pcld_coarse_index_list, mask_point_clouds, frame_list, args):
    segment_global_frame_id_list = torch.where(segment.visible_frame)[0].cpu().numpy()
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

        mask_vertex_idx = list(mask_point_clouds[f'{frame_id}_{mask_id}'])
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


def export_objects(dataset, segment_list, mask_point_clouds, scene_points, coarse_point_frame_matrix, frame_list, args):
    if args.debug:
        print('start exporting')
    total_pcld_list = []
    total_pcld_index_list = []
    total_pcld_bbox_list = []
    total_object_mask_list = []
    
    for i, segment in enumerate(segment_list):
        if len(segment.mask_list) < 2:
            continue
        
        pcld, complete_index_list = segment.get_complete_pcld(scene_points)
        object_pcld_list, object_pcld_coarse_index_list = dbscan_process(pcld, complete_index_list, args.dbscan_threshold)

        object_pcld_list, object_pcld_coarse_index_list, object_bbox_list, object_mask_list = point_filter_and_coverage_computing(coarse_point_frame_matrix, segment, object_pcld_list, object_pcld_coarse_index_list, mask_point_clouds, frame_list, args)

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
    os.makedirs(os.path.join(dataset.object_dict_dir, args.config), exist_ok=True)
    np.save(os.path.join(dataset.object_dict_dir, args.config, 'object_dict.npy'), object_dict, allow_pickle=True)
    return
