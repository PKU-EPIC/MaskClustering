'''
    This script extracts open-vocabulary visual features for each mask following OpenMask3D.
    For each mask, we crop the image with CROP_SCALES=3 scales based on the mask.
    Then we extract the visual features using CLIP model and average these features as the mask feature.
'''

import open_clip
import os
from PIL import Image
import numpy as np
import torch
from utils.config import get_args, get_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

CROP_SCALES = 3 # follow OpenMask3D
args = get_args()

class CroppedImageDataset(Dataset):
    def __init__(self, seq_name_list, frame_id_list, mask_id_list, rgb_path_list, segmentation_path_list, preprocess):
        '''
            Given a list of masks, we calculate the open-vocabulary features for each mask.

            Args:
                seq_name_list: sequence name for each mask
                frame_id_list: frame id for each mask
                mask_id_list: mask id for each mask
                rgb_path_list: rgb path for each mask
                segmentation_path_list: segmentation path for each mask
                preprocess: image preprocessing function
        '''
        self.seq_name_list = seq_name_list
        self.frame_id_list = frame_id_list
        self.mask_id_list = mask_id_list
        self.preprocess = preprocess
        self.rgb_path_list = rgb_path_list
        self.segmentation_path_list = segmentation_path_list

    def __len__(self):
        return len(self.mask_id_list)

    def __getitem__(self, idx):
        def get_cropped_image(mask, rgb):
            '''
                Given a mask and an rgb image, we crop the image with CROP_SCALES scales based on the mask.
            '''
            def mask2box_multi_level(mask, level, expansion_ratio):
                pos = np.where(mask)
                top = np.min(pos[0])
                bottom = np.max(pos[0])
                left = np.min(pos[1])
                right = np.max(pos[1])

                if level == 0:
                    return left, top, right , bottom
                shape = mask.shape
                x_exp = int(abs(right - left)*expansion_ratio) * level
                y_exp = int(abs(bottom - top)*expansion_ratio) * level
                return max(0, left - x_exp), max(0, top - y_exp), min(shape[1], right + x_exp), min(shape[0], bottom + y_exp)

            def crop_image(rgb, mask):
                multiscale_cropped_images = []
                for level in range(CROP_SCALES):
                    left, top, right, bottom = mask2box_multi_level(mask, level, 0.1)
                    cropped_image = rgb[top:bottom, left:right].copy()
                    multiscale_cropped_images.append(cropped_image)
                return multiscale_cropped_images

            mask = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            multiscale_cropped_images = crop_image(rgb, mask)
            return multiscale_cropped_images
        
        def pad_into_square(image):
            width, height = image.size
            new_size = max(width, height)
            new_image = Image.new("RGB", (new_size, new_size), (255,255,255))
            left = (new_size - width) // 2
            top = (new_size - height) // 2
            new_image.paste(image, (left, top))
            return new_image

        seq_name = self.seq_name_list[idx]
        frame_id = self.frame_id_list[idx]
        mask_id = self.mask_id_list[idx]
        rgb_path = self.rgb_path_list[idx]
        segmentation_path = self.segmentation_path_list[idx]

        rgb_image = cv2.imread(rgb_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        segmentation_image = cv2.imread(segmentation_path, cv2.IMREAD_UNCHANGED)
        mask = (segmentation_image == mask_id)
        cropped_images = get_cropped_image(mask, np.array(rgb_image))

        input_images = [self.preprocess(pad_into_square(Image.fromarray(cropped_image))) for cropped_image in cropped_images]
        input_images = torch.stack(input_images)
        return input_images, seq_name, frame_id, mask_id

def load_clip():
    print(f'[INFO] loading CLIP model...')
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
    model.cuda()
    model.eval()
    print(f'[INFO]', ' finish loading CLIP model...')
    return model, preprocess

def main():
    model, preprocess = load_clip()

    seq_name_list, frame_id_list, mask_id_list, rgb_path_list, segmentation_path_list = [], [], [], [], []
    feature_dict = {}
    for seq_name in args.seq_name_list.split('+'):
        args.seq_name = seq_name
        dataset = get_dataset(args)
        object_dict = np.load(f'{dataset.object_dict_dir}/{args.config}/object_dict.npy', allow_pickle=True).item()
        for key, value in object_dict.items():
            mask_list = value['repre_mask_list']
            if len(mask_list) == 0:
                continue
            for mask_info in mask_list:
                seq_name_list.append(seq_name)
                frame_id = mask_info[0]
                frame_id_list.append(frame_id)
                mask_id_list.append(mask_info[1])
                rgb_path, segmentation_path = dataset.get_frame_path(frame_id)
                rgb_path_list.append(rgb_path)
                segmentation_path_list.append(segmentation_path)
        feature_dict[seq_name] = {}

    dataloader = DataLoader(CroppedImageDataset(seq_name_list, frame_id_list, mask_id_list, rgb_path_list, segmentation_path_list, preprocess), batch_size=64, shuffle=False, num_workers=16)
    
    print('[INFO] extracting features')
    for images, seq_names, frame_ids, mask_ids in tqdm(dataloader):
        images = images.reshape(-1, 3, 224, 224)
        image_input = images.cuda()
        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().numpy()
        for i in range(len(image_features) // CROP_SCALES):
            feature_dict[seq_names[i]][f'{frame_ids[i]}_{mask_ids[i]}'] = image_features[CROP_SCALES*i:CROP_SCALES*(i+1)].mean(axis=0)
    print('[INFO] finish extracting features')

    for seq_name in args.seq_name_list.split('+'):
        args.seq_name = seq_name
        dataset = get_dataset(args)
        np.save(os.path.join(dataset.object_dict_dir, f'{args.config}/open-vocabulary_features.npy'), feature_dict[seq_name])

if __name__ == '__main__':
    main()