import open3d as o3d
import numpy as np
import os
import cv2
from evaluation.constants import MATTERPORT_LABELS, MATTERPORT_IDS

class MatterportDataset:
    def __init__(self, seq_name) -> None:
        self.seq_name = seq_name
        self.root = f'./data/matterport3d/scans/{seq_name}/{seq_name}'
        self.rgb_dir = f'{self.root}/undistorted_color_images'
        self.depth_dir = f'{self.root}/undistorted_depth_images'
        self.cam_param_dir = f'{self.root}/undistorted_camera_parameters/{seq_name}.conf'
        self.point_cloud_path = f'{self.root}/house_segmentations/{seq_name}.ply'
        self.mesh_path = self.point_cloud_path
        self.rgb_names, self.depth_names, self.intrinsics, self.extrinsics = \
            self._obtain_intr_extr()
        
        # output
        self.segmentation_dir = f'{self.root}/output/mask/'
        self.object_dict_dir = f'{self.root}/output/object'

        self.depth_scale = 4000.0 # (0.25mm per unit) 1u = 1/4000 m
        self.image_size = (1280, 1024)

    
    def get_frame_list(self, step):
        image_list = [os.path.join(self.rgb_dir, rgb_name) for rgb_name in self.rgb_names]

        end = len(image_list)
        frame_id_list = np.arange(0, end, step)
        return list(frame_id_list)
    

    def _obtain_intr_extr(self):
        '''Obtain the intrinsic and extrinsic parameters of Matterport3D.'''
       
        with open(self.cam_param_dir, 'r') as file:
            lines = file.readlines()

        def remove_items(test_list, item):
            return [i for i in test_list if i != item]

        intrinsics = []
        extrinsics = []
        img_names = []
        depth_names = []
        for i, line in enumerate(lines):
            line = line.strip()
            if 'intrinsics_matrix' in line:
                line = line.replace('intrinsics_matrix ', '')
                line = line.split(' ')
                line = remove_items(line, '')
                if len(line) !=9:
                    print('[WARN] something wrong at {}'.format(i))
                intrinsic = np.asarray(line).astype(float).reshape(3, 3)
                intrinsics.extend([intrinsic, intrinsic, intrinsic, intrinsic, intrinsic, intrinsic])
            elif 'scan' in line:
                line = line.split(' ')
                img_names.append(line[2])
                depth_names.append(line[1])

                line = remove_items(line, '')[3:]
                if len(line) != 16:
                    print('[WARN] something wrong at {}'.format(i))
                extrinsic = np.asarray(line).astype(float).reshape(4, 4)  
                extrinsic[:3, 1] *= -1.0 # gl2cv
                extrinsic[:3, 2] *= -1.0
                extrinsics.append(extrinsic)

        intrinsics = np.stack(intrinsics, axis=0)
        extrinsics = np.stack(extrinsics, axis=0)
        img_names = np.asarray(img_names)

        return img_names, depth_names, intrinsics, extrinsics


    def get_intrinsics(self, frame_id):
        K = self.intrinsics[frame_id]
        intrinisc_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
        intrinisc_cam_parameters.set_intrinsics(self.image_size[0], self.image_size[1], K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        return intrinisc_cam_parameters
    

    def get_extrinsic(self, frame_id):
        return self.extrinsics[frame_id] 
    

    def get_depth(self, frame_id):
        depth_path = os.path.join(self.depth_dir, self.depth_names[frame_id])
        depth = cv2.imread(depth_path, -1).astype(np.uint16)
        depth = depth / self.depth_scale
        depth = depth.astype(np.float32)
        return depth


    def get_rgb(self, frame_id, change_color=True):
        rgb = cv2.imread(os.path.join(self.rgb_dir, self.rgb_names[frame_id]))
        if change_color:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb    


    def get_segmentation(self, frame_id, align_with_depth=False):
        frame_name = self.rgb_names[frame_id][:-4]
        segmentation_path = os.path.join(self.segmentation_dir, f'{frame_name}.png')
        if not os.path.exists(segmentation_path):
            assert False, f"Segmentation not found: {segmentation_path}"
        segmentation = cv2.imread(segmentation_path, cv2.IMREAD_UNCHANGED)
        return segmentation


    def get_frame_path(self, frame_id):
        rgb_path = os.path.join(self.rgb_dir, self.rgb_names[frame_id])
        frame_name = self.rgb_names[frame_id][:-4]
        segmentation_path = os.path.join(self.segmentation_dir, f'{frame_name}.png')
        return rgb_path, segmentation_path


    def get_label_features(self):
        label_features_dict = np.load(f'data/text_features/matterport3d.npy', allow_pickle=True).item()
        return label_features_dict


    def get_scene_points(self):
        mesh = o3d.io.read_point_cloud(self.point_cloud_path)
        vertices = np.asarray(mesh.points)
        return vertices


    def get_label_id(self):
        self.label2id = {}
        self.id2label = {}
        for label, id in zip(MATTERPORT_LABELS, MATTERPORT_IDS):
            self.label2id[label] = id
            self.id2label[id] = label
        return self.label2id, self.id2label