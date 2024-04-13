import os
import numpy as np
from plyfile import PlyData
import tqdm
import json
from constants import RAW_TO_NYU, MATTERPORT_VALID_IDS

raw_data_dir = '../../data/matterport3d/scans'
gt_dir = '../../data/matterport3d/gt'
split_file_path = '../../splits/matterport3d.txt'

def load_instance_json(filepath):
    with open(filepath, 'r') as f:
        info = json.load(f)
    instances = info["segGroups"]
    instance_segments_id = []
    for instance in instances:
        instance_segments_id.append(np.array(instance["segments"]))
    return instance_segments_id

def load_face_json(filepath):
    with open(filepath, 'r') as f:
        info = json.load(f)
    return np.array(info["segIndices"])

def load_ply(filepath):
    with open(filepath, "rb") as f:
        plydata = PlyData.read(f)
    vdata = plydata.elements[0].data # vertex
    coords = np.array([vdata["x"], vdata["y"], vdata["z"]], dtype=np.float32).T
       
    fdata = plydata['face'].data
    faces = np.stack(fdata['vertex_indices'])

    face_semantic_id = np.array(fdata["category_id"], dtype=np.int32)
    vert_semantic_id = np.zeros(coords.shape[0], dtype=np.int32)
    vert_semantic_id[faces.reshape(-1)] = face_semantic_id[None].repeat(3, axis=1).reshape(-1)
    
    return vert_semantic_id, faces

def convert_gt(root_dir, seq_name, output_dir):
    scene_dir = os.path.join(root_dir, seq_name, seq_name)
    vert_semantic_id, faces = load_ply(os.path.join(scene_dir, 'house_segmentations', f'{seq_name}.ply'))
    
    face_segment_id = load_face_json(os.path.join(scene_dir, 'house_segmentations', f'{seq_name}.fsegs.json'))
    vert_segment_id = np.zeros_like(vert_semantic_id)
    vert_segment_id[faces.reshape(-1)] = face_segment_id[None].repeat(3, axis=1).reshape(-1)

    segment_ids = np.unique(vert_segment_id)
    instance_segments_id = load_instance_json(os.path.join(scene_dir, 'house_segmentations', f'{seq_name}.semseg.json'))
    segment_instance_id = np.full(segment_ids.max() + 1, -1)
    for instance_id, segments_id in enumerate(instance_segments_id):
        segment_instance_id[segments_id] = instance_id
    
    vert_instance_id = segment_instance_id[vert_segment_id]

    assert vert_instance_id.shape == vert_segment_id.shape
    assert vert_instance_id.min() >= 0 and vert_instance_id.max() <= len(instance_segments_id)

    vert_semantic_id[vert_semantic_id < 0] = 0
    vert_semantic_id = RAW_TO_NYU[vert_semantic_id]
    vert_semantic_id[np.isin(vert_semantic_id, MATTERPORT_VALID_IDS, invert=True)] = 0

    print(np.unique(vert_semantic_id).shape[0], ' semantics, ', len(instance_segments_id), ' instances')
    gt_id = vert_semantic_id * 1000 + vert_instance_id + 1

    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, f'{seq_name}.txt'), gt_id.astype(np.int32), fmt="%d")

if __name__ == "__main__":
    with open(split_file_path, 'r') as f:
        seq_name_list = f.readlines()
    seq_name_list = [seq_name.strip() for seq_name in seq_name_list]
    for seq_name in tqdm.tqdm(seq_name_list):
        convert_gt(raw_data_dir, seq_name, gt_dir)