from utils.config import get_args, get_dataset
import os
import cv2
import numpy as np

def create_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap

def main(dataset, vis_dir, frame_id):
    segmentation_image = dataset.get_segmentation(frame_id)
    color_segmentation = np.zeros((segmentation_image.shape[0], segmentation_image.shape[1], 3), dtype=np.uint8)
    mask_ids = np.unique(segmentation_image)
    mask_ids.sort()

    text_list, text_center_list = [], []
    for mask_id in mask_ids:
        if mask_id == 0:
            continue
        color_segmentation[segmentation_image == mask_id] = colormap[mask_id]
        mask_pos = np.where(segmentation_image == mask_id)
        mask_center = (int(np.mean(mask_pos[1])), int(np.mean(mask_pos[0])))
        text_list.append(str(mask_id))
        text_center_list.append(mask_center)
    
    for text, text_center in zip(text_list, text_center_list):
        cv2.putText(color_segmentation, text, text_center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    raw_rgb = dataset.get_rgb(frame_id, change_color=False)
    concatenate_image = np.concatenate((raw_rgb, color_segmentation), axis=1)
    concatenate_image = cv2.resize(concatenate_image, (concatenate_image.shape[1] // 2, concatenate_image.shape[0] // 2))
    cv2.imwrite(os.path.join(vis_dir, str(frame_id) + '.png'), concatenate_image)
    return

if __name__ == '__main__':
    args = get_args()
    dataset = get_dataset(args)
    colormap = create_colormap()
    vis_dir = os.path.join(dataset.segmentation_dir, '..', 'vis_mask')
    os.makedirs(vis_dir, exist_ok=True)
    frame_list = dataset.get_frame_list(args.step)
    for frame_id in frame_list:
        main(dataset, vis_dir, frame_id)
        break