'''
    This script is used to generate the semantic labels for the objects in the scene.
'''
from utils.config import get_dataset, get_args
import os
import numpy as np

def main(args):
    dataset = get_dataset(args)
    total_point_num = dataset.get_scene_points().shape[0]

    label_features_dict = dataset.get_label_features()
    label_text_features = np.stack(list(label_features_dict.values()))
    descriptions = list(label_features_dict.keys())
    
    object_dict = np.load(f'{dataset.object_dict_dir}/{args.config}/object_dict.npy', allow_pickle=True).item()
    clip_feature = np.load(f'{dataset.object_dict_dir}/{args.config}/open-vocabulary_features.npy', allow_pickle=True).item()
    label2id = dataset.get_label_id()[0]

    pred_dir = os.path.join('data/prediction', args.config)
    os.makedirs(pred_dir, exist_ok=True)

    num_instance = len(object_dict)
    pred_dict = {
        "pred_masks": np.zeros((total_point_num, num_instance), dtype=bool), 
        "pred_score":  np.ones(num_instance),
        "pred_classes" : np.zeros(num_instance, dtype=np.int32)
    }

    # For each object, average the visual features of the representative masks as its object feature. 
    # Then get semantic label according to the similarity between the object feature and the label text features.
    for idx, (key, value) in enumerate(object_dict.items()):
        repre_mask_list = value['repre_mask_list']
        if len(repre_mask_list) == 0:
            continue

        feature_list = []
        feature_list = [clip_feature[f'{mask_info[0]}_{mask_info[1]}'] for mask_info in repre_mask_list]
        feature = np.stack(feature_list)
        object_feature = np.mean(feature, axis=0, keepdims=True)

        raw_similarity = np.dot(object_feature, label_text_features.T)
        exp_sim = np.exp(raw_similarity * 100)
        prob = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)
        max_label_id = np.argmax(np.max(prob, axis=0))

        label_id = label2id[descriptions[max_label_id]]
        pred_dict['pred_classes'][idx] = label_id

        point_ids = object_dict[key]['point_ids']
        binary_mask = np.zeros(total_point_num, dtype=bool)
        binary_mask[list(point_ids)] = True
        pred_dict['pred_masks'][:, idx] = binary_mask

    np.savez(f'{pred_dir}/{args.seq_name}.npz', **pred_dict) 

if __name__ == '__main__':
    args = get_args()
    main(args)