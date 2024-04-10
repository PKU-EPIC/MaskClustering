import open3d as o3d
import numpy as np
import os
import cv2
import json
import collections

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
BaseCamera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

    @property
    def world_to_camera(self) -> np.ndarray:
        R = qvec2rotmat(self.qvec)
        t = self.tvec
        world2cam = np.eye(4)
        world2cam[:3, :3] = R
        world2cam[:3, 3] = t
        return world2cam

class Camera(BaseCamera):
    @property
    def K(self):
        K = np.eye(3)
        if self.model == "SIMPLE_PINHOLE" or self.model == "SIMPLE_RADIAL" or self.model == "RADIAL" or self.model == "SIMPLE_RADIAL_FISHEYE" or self.model == "RADIAL_FISHEYE":
            K[0, 0] = self.params[0]
            K[1, 1] = self.params[0]
            K[0, 2] = self.params[1]
            K[1, 2] = self.params[2]
        elif self.model == "PINHOLE" or self.model == "OPENCV" or self.model == "OPENCV_FISHEYE" or self.model == "FULL_OPENCV" or self.model == "FOV" or self.model == "THIN_PRISM_FISHEYE":
            K[0, 0] = self.params[0]
            K[1, 1] = self.params[1]
            K[0, 2] = self.params[2]
            K[1, 2] = self.params[3]
        else:
            raise NotImplementedError
        return K

def read_images_text(path):
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                    tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images

def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

class ScanNetPPDataset:
    def __init__(self, seq_name) -> None:
        self.seq_name = seq_name
        self.root = f'./data/scannetpp/data/{seq_name}'
        self.rgb_dir = f'{self.root}/iphone/rgb'
        self.depth_dir = f'{self.root}/iphone/render_depth'

        self.mask_dir = f'{self.root}/output/mask'
        self.object_dict_dir = f'{self.root}/output/object'
        self.mask_image_dir = f'{self.mask_dir}/image'

        self.mesh_path = f'data/scannetpp/pcld/{seq_name}/sampled_025.ply'
        self.pred_dir = f'data/scannetpp/evaluation/pred'

        self.load_meta_data()

        self.depth_scale = 1000.0
        self.roof_height = 20
        self.image_size = (1920, 1440)

    def load_meta_data(self):
        self.frame_id_list = []
        
        cameras = read_cameras_text(os.path.join(self.root, 'iphone/colmap', "cameras.txt"))
        images = read_images_text(os.path.join(self.root, 'iphone/colmap', "images.txt"))
        camera = next(iter(cameras.values()))
        fx, fy, cx, cy = camera.params[:4]
        intrinsics = {}
        extrinsics = {}

        for _, image in (images.items()):
            image_id = int(image.name.split('.')[0].split('_')[1])
            self.frame_id_list.append(image_id)
            world_to_camera = image.world_to_camera
            extrinsics[image_id] = np.linalg.inv(world_to_camera)
            intrinsics[image_id] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        self.extrinsics = extrinsics
        self.intrinsics = intrinsics
    
    def get_video_end(self):
        return self.frame_id_list[-1]
    
    def get_frame_list(self, start, end, stride):
        # depth_list = os.listdir(self.depth_dir)
        # frame_id_list = [int(depth_name.split('.')[0].split('_')[1]) for depth_name in depth_list]
        # frame_id_list.sort()
        # return frame_id_list
        return self.frame_id_list
    
    def get_intrinsics(self, frame_id):
        intrinsic_matrix = self.intrinsics[frame_id]

        intrinisc_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
        intrinisc_cam_parameters.set_intrinsics(self.image_size[0], self.image_size[1], intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2])
        return intrinisc_cam_parameters
    
    def get_extrinsic(self, frame_id):
        return self.extrinsics[frame_id]
    
    def get_depth(self, frame_id):
        depth_path = os.path.join(self.depth_dir, 'frame_%06d.png' % frame_id)
        depth = cv2.imread(depth_path, -1)
        depth = depth / self.depth_scale
        depth = depth.astype(np.float32)
        return depth

    def get_rgb(self, frame_id, change_color=True, orginal_size=False):
        rgb_path = os.path.join(self.rgb_dir, 'frame_%06d.jpg' % frame_id)
        rgb = cv2.imread(rgb_path)
        if change_color:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        # if not orginal_size:
        #     rgb = cv2.resize(rgb, (256, 192))
        return rgb    

    def get_total_vertex_num(self):
        mesh = o3d.io.read_point_cloud(self.mesh_path)
        vertices = np.asarray(mesh.points)
        return len(vertices)
    
    def get_frame_path(self, frame_id):
        rgb_path = os.path.join(self.rgb_dir, 'frame_%06d.jpg' % frame_id)
        segmentation_path = os.path.join(self.mask_image_dir, f'{frame_id}.npy')
        return rgb_path, segmentation_path
    
    def get_label_features(self):
        label_features_dict = np.load(f'data/text_features/scannetpp.npy', allow_pickle=True).item()
        return label_features_dict

    def get_label_id(self):
        with open('data/scannetpp/metadata/class_id.txt', 'r') as fp:
            self.class_id = [int(x) for x in fp.read().splitlines()]

        with open('data/scannetpp/metadata/class_name.txt', 'r') as fp:
            self.class_label = [x for x in fp.read().splitlines()]
        
        self.label2id = {}
        for label, id in zip(self.class_label, self.class_id):
            self.label2id[label] = id

        self.id2label = {}
        for label, id in zip(self.class_label, self.class_id):
            self.id2label[id] = label

        return self.label2id, self.id2label

    def get_label_color(self):
        raise NotImplementedError