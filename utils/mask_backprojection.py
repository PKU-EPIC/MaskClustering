import numpy as np
from pytorch3d.ops import ball_query
import torch
import open3d as o3d
from utils.geometry import denoise
from torch.nn.utils.rnn import pad_sequence

COVERAGE_THRESHOLD = 0.3
DISTANCE_THRESHOLD = 0.03
FEW_POINTS_THRESHOLD = 25
DEPTH_TRUNC = 20
BBOX_EXPAND = 0.1


def backproject(depth, intrinisc_cam_parameters, extrinsics):
    """
    convert color and depth to view pointcloud
    """
    depth = o3d.geometry.Image(depth)
    pcld = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinisc_cam_parameters, depth_scale=1, depth_trunc=DEPTH_TRUNC)
    pcld.transform(extrinsics)
    return pcld


def get_neighbor(valid_points, scene_points, lengths_1, lengths_2):
    _, neighbor_in_scene_pcld, _ = ball_query(valid_points, scene_points, lengths_1, lengths_2, K=20, radius=DISTANCE_THRESHOLD, return_nn=False)
    return neighbor_in_scene_pcld


def get_depth_mask(depth):
    depth_tensor = torch.from_numpy(depth).cuda()
    depth_mask = torch.logical_and(depth_tensor > 0, depth_tensor < DEPTH_TRUNC).reshape(-1)
    return depth_mask


def crop_scene_points(mask_points, scene_points):
    x_min, x_max = torch.min(mask_points[:, 0]), torch.max(mask_points[:, 0])
    y_min, y_max = torch.min(mask_points[:, 1]), torch.max(mask_points[:, 1])
    z_min, z_max = torch.min(mask_points[:, 2]), torch.max(mask_points[:, 2])

    selected_point_mask = (scene_points[:, 0] > x_min) & (scene_points[:, 0] < x_max) & (scene_points[:, 1] > y_min) & (scene_points[:, 1] < y_max) & (scene_points[:, 2] > z_min) & (scene_points[:, 2] < z_max)
    selected_point_ids = torch.where(selected_point_mask)[0]
    cropped_scene_points = scene_points[selected_point_ids]
    return cropped_scene_points, selected_point_ids


def turn_mask_to_point(dataset, scene_points, mask_image, frame_id):
    intrinisc_cam_parameters = dataset.get_intrinsics(frame_id)
    extrinsics = dataset.get_extrinsic(frame_id)
    if np.sum(np.isinf(extrinsics)) > 0:
        return {}, [], set()

    mask_image = torch.from_numpy(mask_image).cuda().reshape(-1)
    ids = torch.unique(mask_image).cpu().numpy()
    ids.sort()
    
    depth = dataset.get_depth(frame_id)
    depth_mask = get_depth_mask(depth)

    colored_pcld = backproject(depth, intrinisc_cam_parameters, extrinsics)
    view_points = np.asarray(colored_pcld.points)

    mask_points_list = []
    mask_points_num_list = []
    scene_points_list = []
    scene_points_num_list = []
    selected_point_ids_list = []
    initial_valid_mask_ids = []
    for mask_id in ids:
        if mask_id == 0:
            continue
        segmentation = mask_image == mask_id
        valid_mask = segmentation[depth_mask].cpu().numpy()

        mask_pcld = o3d.geometry.PointCloud()
        mask_points = view_points[valid_mask]
        if len(mask_points) < FEW_POINTS_THRESHOLD:
            continue
        mask_pcld.points = o3d.utility.Vector3dVector(mask_points)

        mask_pcld = mask_pcld.voxel_down_sample(voxel_size=DISTANCE_THRESHOLD)
        mask_pcld, _ = denoise(mask_pcld)
        mask_points = np.asarray(mask_pcld.points)
        
        if len(mask_points) < FEW_POINTS_THRESHOLD:
            continue
        
        mask_points = torch.tensor(mask_points).float().cuda()
        cropped_scene_points, selected_point_ids = crop_scene_points(mask_points, scene_points)
        initial_valid_mask_ids.append(mask_id)
        mask_points_list.append(mask_points)
        scene_points_list.append(cropped_scene_points)
        mask_points_num_list.append(len(mask_points))
        scene_points_num_list.append(len(cropped_scene_points))
        selected_point_ids_list.append(selected_point_ids)

    if len(initial_valid_mask_ids) == 0:
        return {}, [], []
    mask_points_tensor = pad_sequence(mask_points_list, batch_first=True, padding_value=0)
    scene_points_tensor = pad_sequence(scene_points_list, batch_first=True, padding_value=0)

    lengths_1 = torch.tensor(mask_points_num_list).cuda()
    lengths_2 = torch.tensor(scene_points_num_list).cuda()
    neighbor_in_scene_pcld = get_neighbor(mask_points_tensor, scene_points_tensor, lengths_1, lengths_2)

    valid_mask_ids = []
    mask_info = {}
    frame_point_ids = set()

    for i, mask_id in enumerate(initial_valid_mask_ids):
        mask_neighbor = neighbor_in_scene_pcld[i] # P, 20
        mask_point_num = mask_points_num_list[i] # Pi
        mask_neighbor = mask_neighbor[:mask_point_num] # Pi, 20

        valid_neighbor = mask_neighbor != -1 # Pi, 20
        neighbor = torch.unique(mask_neighbor[valid_neighbor])
        neighbor_in_complete_scene_points = selected_point_ids_list[i][neighbor].cpu().numpy()
        coverage = torch.any(valid_neighbor, dim=1).sum().item() / mask_point_num

        if coverage < COVERAGE_THRESHOLD:
            continue
        valid_mask_ids.append(mask_id)
        mask_info[mask_id] = set(neighbor_in_complete_scene_points)
        frame_point_ids.update(mask_info[mask_id])

    return mask_info, valid_mask_ids, list(frame_point_ids)


def frame_backprojection(dataset, scene_points, frame_id):
    mask_image = dataset.get_segmentation(frame_id, align_with_depth=True)
    mask_info, _, frame_point_ids = turn_mask_to_point(dataset, scene_points, mask_image, frame_id)
    return mask_info, frame_point_ids