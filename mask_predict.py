# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
import cv2
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import warnings
import numpy as np
from tqdm import tqdm
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--seq_name_list",
        type=str
    )
    parser.add_argument(
        "--root",
        type=str
    )
    parser.add_argument(
        "--image_path_pattern",
        type=str
    )
    parser.add_argument(
        "--dataset",
        type=str
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    seq_name_list = args.seq_name_list.split('+')
    for i, seq_name in tqdm(enumerate(seq_name_list), total=len(seq_name_list)):
        seq_dir = os.path.join(args.root, seq_name)
        image_list = sorted(glob.glob(os.path.join(seq_dir, args.image_path_pattern)))
        output_dir = os.path.join(seq_dir, seq_name, 'output/mask') if args.dataset == 'matterport3d' else os.path.join(seq_dir, 'output/mask')
        
        os.makedirs(output_dir, exist_ok=True)
        
        for path in (image_list):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            predictions = demo.run_on_image(img)

            ##### color_mask
            pred_masks = predictions["instances"].pred_masks
            pred_scores = predictions["instances"].scores
            
            # select by confidence threshold
            selected_indexes = (pred_scores >= args.confidence_threshold)
            selected_scores = pred_scores[selected_indexes]
            selected_masks  = pred_masks[selected_indexes]
            _, m_H, m_W = selected_masks.shape
            mask_image = np.zeros((m_H, m_W), dtype=np.uint8)

            # rank
            mask_id = 1
            selected_scores, ranks = torch.sort(selected_scores)
            for index in ranks:
                num_pixels = torch.sum(selected_masks[index])
                if num_pixels < 400:
                    # ignore small masks
                    continue
                mask_image[(selected_masks[index]==1).cpu().numpy()] = mask_id
                mask_id += 1
            cv2.imwrite(os.path.join(output_dir, os.path.basename(path).split('.')[0] + '.png'), mask_image)