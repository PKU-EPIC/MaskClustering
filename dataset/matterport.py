import open3d as o3d
import numpy as np
import os
import cv2

class MatterportDataset:
    def __init__(self, seq_name) -> None:
        self.seq_name = seq_name
        self.root = f'./data/matterport3d/v1/scans/{seq_name}/{seq_name}'
        self.rgb_dir = f'{self.root}/undistorted_color_images'
        self.depth_dir = f'{self.root}/undistorted_depth_images'
        self.cam_param_dir = f'{self.root}/undistorted_camera_parameters/{seq_name}.conf'
        self.mesh_path = f'{self.root}/house_segmentations/{seq_name}.ply'
        self.rgb_names, self.depth_names, self.intrinsics, self.extrinsics = \
            self._obtain_intr_extr_matterport()
        self.pred_dir = f'data/matterport3d/instance_segmentation/pred'
        
        # output
        self.output_root = f'{self.root}/output'
        self.mask_image_dir = f'{self.output_root}/mask/{mask_generator}/image'
        self.object_dict_dir = f'{self.output_root}/object/dict'

        self.depth_scale = 4000.0 # (0.25mm per unit) 1u = 1/4000 m
        self.image_size = (1280, 1024)

    def get_video_end(self): #TODO
        video_end = len(self.get_image_list()) - 1
        return video_end
    
    def get_window_list(self): #TODO
        video_start = 0
        video_end = self.get_video_end()
        window_list = []
        start_list = list(range(video_start, video_end, 100))
        for window_start in start_list:
            window_end = min(window_start+200, video_end)
            window_list.append((window_start, window_end))
        return window_list

    def get_image_list(self, stride=1):
        image_list = [os.path.join(self.rgb_dir, rgb_name) for rgb_name in self.rgb_names]
        image_list = image_list[::stride]
        return image_list
    
    def get_frame_list(self, start, end, stride):
        if end < 0: # full video
            end = self.get_video_end() + 1
        else:
            end = min(self.get_video_end() + 1, end)
        frame_id_list = np.arange(start, end, stride)
        return frame_id_list
    
    def _obtain_intr_extr_matterport(self):
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
        assert depth.min() >= 0 and depth.max() < 15, f"{depth.min()} {depth.max()}"
        return depth

    def get_rgb(self, frame_id, change_color=True, orginal_size=True):
        rgb = cv2.imread(os.path.join(self.rgb_dir, self.rgb_names[frame_id]))
        if change_color:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb    

    def get_frame_path(self, frame_id):
        rgb_path = os.path.join(self.rgb_dir, self.rgb_names[frame_id])
        segmentation_path = os.path.join(self.mask_image_dir, f'{frame_id}.npy')
        return rgb_path, segmentation_path

    def get_label_features(self):
        label_features_dict = np.load(f'data/text_features/matterport3d.npy', allow_pickle=True).item()
        return label_features_dict

    def get_total_vertex_num(self):
        mesh = o3d.io.read_triangle_mesh(self.mesh_path)
        vertices = np.asarray(mesh.vertices)
        return len(vertices)
    
    def get_roof_height(self):
        mesh = o3d.io.read_triangle_mesh(self.mesh_path)
        vertices = np.asarray(mesh.vertices)
        roof_height = np.max(vertices[:, 2])
        return roof_height

    def get_gt_labels(self):
        raise NotImplementedError
    
    def get_label_id(self):
        self.label2id = {}
        self.id2label = {}
        for label, id in zip(MATTERPORT_LABELS_160, MATTERPORT_VALID_IDS):
            self.label2id[label] = id
            self.id2label[id] = label
        return self.label2id, self.id2label

    def get_label_color(self):
        raise NotImplementedError