from utils.config import get_dataset, get_args
import os
import numpy as np

def main(args):
    dataset = get_dataset(args)
    total_vertex_num = dataset.get_total_vertex_num()

    if args.dataset_type == 'scannet':
        label_text_features_dict = np.load(f'data/text_features/scannet.npy', allow_pickle=True).item()
    elif args.dataset_type == 'matterport3d':
        label_text_features_dict = np.load(f'data/matterport3d/text_features_160.npy', allow_pickle=True).item()
    elif args.dataset_type == 'scannetpp':
        label_text_features_dict = np.load(f'data/scannetpp/text_features_1500.npy', allow_pickle=True).item()
    label_text_features = np.stack(list(label_text_features_dict.values()))
    descriptions = list(label_text_features_dict.keys())
    
    object_dict = np.load(f'{dataset.object_dict_dir}/{args.config_type}/object_dict.npy', allow_pickle=True).item()
    clip_feature = np.load(f'{dataset.object_dict_dir}/object_open-vocabulary_features.npy', allow_pickle=True).item()
    label2id = dataset.get_label_id()[0]

    pred_dir = os.path.join(dataset.pred_dir, args.config_type)
    os.makedirs(pred_dir, exist_ok=True)

    num_instance = len(object_dict)
    pred_dict = {
        "pred_masks": np.zeros((total_vertex_num, num_instance), dtype=bool), 
        "pred_score":  np.ones(num_instance),
        "pred_classes" : np.zeros(num_instance, dtype=np.int32)
    }

    for idx, (key, value) in enumerate(object_dict.items()):
        mask_info_list = value['mask_list']
        if len(mask_info_list) == 0:
            continue
        represent_mask_list = value['repre_mask_list']

        feature_list = []
        feature_list = [clip_feature[f'{mask_info[0]}_{mask_info[1]}'] for mask_info in represent_mask_list]
        feature = np.stack(feature_list)
        feature = np.mean(feature, axis=0, keepdims=True)

        raw_similarity = np.dot(feature, label_text_features.T)
        exp_sim = np.exp(raw_similarity * 100)
        prob = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)
        max_label_id = np.argmax(np.max(prob, axis=0))

        label_id = label2id[descriptions[max_label_id]]
        pred_dict['pred_classes'][idx] = label_id

        vertex_index = object_dict[key]['vertex_index']
        binary_mask = np.zeros(total_vertex_num, dtype=bool)
        binary_mask[list(vertex_index)] = True
        pred_dict['pred_masks'][:, idx] = binary_mask

    np.savez(f'{pred_dir}/{args.seq_name}.npz', **pred_dict) 

if __name__ == '__main__':
    try:
        args = get_args()
        main(args)
    except:
        # show error
        import traceback
        traceback.print_exc()
        
        with open('failed.txt', 'a') as f:
            f.write(args.seq_name + '\n')