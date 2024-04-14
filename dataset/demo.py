import open3d as o3d
import numpy as np
import os
import cv2
from evaluation.constants import SCANNET_LABELS, SCANNET_IDS

class DemoDataset:

    def __init__(self, seq_name) -> None:
        self.seq_name = seq_name
        self.root = f'./data/demo/{seq_name}'
        self.rgb_dir = f'{self.root}/color_640'
        self.depth_dir = f'{self.root}/depth'
        self.segmentation_dir = f'{self.root}/output/mask'
        self.object_dict_dir = f'{self.root}/output/object'
        self.point_cloud_path = f'{self.root}/{seq_name}_vh_clean_2.ply'
        self.mesh_path = self.point_cloud_path
        self.extrinsics_dir = f'{self.root}/pose'

        self.depth_scale = 1000.0
        self.image_size = (640, 480)
    

    def get_frame_list(self, stride):
        image_list = os.listdir(self.rgb_dir)
        image_list = sorted(image_list, key=lambda x: int(x.split('.')[0]))

        end = int(image_list[-1].split('.')[0]) + 1
        frame_id_list = np.arange(0, end, stride)
        return list(frame_id_list)
    

    def get_intrinsics(self, frame_id):
        intrinsic_path = f'{self.root}/intrinsic_640.txt'
        intrinsics = np.loadtxt(intrinsic_path)

        intrinisc_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
        intrinisc_cam_parameters.set_intrinsics(640, 480, intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
        return intrinisc_cam_parameters
    

    def get_extrinsic(self, frame_id):
        pose_path = os.path.join(self.extrinsics_dir, str(frame_id) + '.txt')
        pose = np.loadtxt(pose_path)
        return pose
    

    def get_depth(self, frame_id):
        depth_path = os.path.join(self.depth_dir, str(frame_id) + '.png')
        depth = cv2.imread(depth_path, -1)
        depth = depth / self.depth_scale
        depth = depth.astype(np.float32)
        return depth


    def get_rgb(self, frame_id, change_color=True):
        rgb_path = os.path.join(self.rgb_dir, str(frame_id) + '.jpg')
        rgb = cv2.imread(rgb_path)

        if change_color:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb    


    def get_segmentation(self, frame_id, align_with_depth=False):
        segmentation_path = os.path.join(self.segmentation_dir, f'{frame_id}.png')
        if not os.path.exists(segmentation_path):
            assert False, f"Segmentation not found: {segmentation_path}"
        segmentation = cv2.imread(segmentation_path, cv2.IMREAD_UNCHANGED)
        return segmentation


    def get_frame_path(self, frame_id):
        rgb_path = os.path.join(self.rgb_dir, str(frame_id) + '.jpg')
        segmentation_path = os.path.join(self.segmentation_dir, f'{frame_id}.png')
        return rgb_path, segmentation_path
    

    def get_label_features(self):
        label_features_dict = np.load(f'data/text_features/scannet.npy', allow_pickle=True).item()
        return label_features_dict


    def get_scene_points(self):
        mesh = o3d.io.read_point_cloud(self.point_cloud_path)
        vertices = np.asarray(mesh.points)
        return vertices
    
    
    def get_label_id(self):
        self.class_id = SCANNET_IDS
        self.class_label = SCANNET_LABELS

        self.label2id = {}
        self.id2label = {}
        for label, id in zip(self.class_label, self.class_id):
            self.label2id[label] = id
            self.id2label[id] = label

        return self.label2id, self.id2label