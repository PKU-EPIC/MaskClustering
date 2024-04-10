import open_clip
from open_clip import tokenizer
import torch
import numpy as np
from evaluation.constants import MATTERPORT_LABELS, SCANNET_LABELS, SCANNETPP_LABELS

def load_clip():
    print(f'[INFO] loading CLIP model...')
    model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
    model.cuda()
    model.eval()
    print(f'[INFO]', ' finish loading CLIP model...')
    return model

def extract_text_feature(save_path, descriptions):
    text_tokens = tokenizer.tokenize(descriptions).cuda()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()

    text_features_dict = {}
    for i, description in enumerate(descriptions):
        text_features_dict[description] = text_features[i]

    np.save(save_path, text_features_dict)

model = load_clip()
extract_text_feature('data/text_features/scannet.npy', SCANNET_LABELS)
extract_text_feature('data/text_features/scannetpp.npy', SCANNETPP_LABELS)
extract_text_feature('data/text_features/matterport3d.npy', MATTERPORT_LABELS)