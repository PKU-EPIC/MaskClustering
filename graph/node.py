import torch
import open3d as o3d

class Node:
    def __init__(self, mask_list, visible_frame, contained_mask, complete_vertex_index, node_info, son_node_info):
        self.mask_list = mask_list
        self.visible_frame = visible_frame # one-hot vector
        self.contained_mask = contained_mask # one-hot vector
        self.complete_vertex_index = complete_vertex_index
        self.node_info = node_info
        self.son_node_info = son_node_info
    
    @ staticmethod
    def create_node_from_list(segment_list, node_info):
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
        return Node(mask_list, frame_one_hot.float(), frame_mask_one_hot.float(), complete_vertex_index, node_info, son_node_info)
    
    def get_complete_pcld(self, scene_points):
        complete_index_list = list(self.complete_vertex_index)
        points = scene_points[complete_index_list]
        pcld = o3d.geometry.PointCloud()
        pcld.points = o3d.utility.Vector3dVector(points)
        return pcld, complete_index_list
