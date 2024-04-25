<p align="center">

  <h1 align="center">MaskClustering: View Consensus based Mask Graph Clustering
for Open-Vocabulary 3D Instance Segmentation</h1>
    <p align="center">
        <a href="https://miyandoris.github.io/">Mi Yan</a><sup>1,2</sup></span>, 
        <a href="https://jzhzhang.github.io/">Jiazhao Zhang</a><sup>1,2</sup>
        <a href="https://github.com/fzy139/">Yan Zhu</a><sup>1</sup>, 
        <a href="https://hughw19.github.io/">He Wang</a><sup>1,2,3</sup>
        <br>
        <sup>1</sup>Peking University, 
        <sup>2</sup>Beijing Academy of Artificial Intelligence, 
        <sup>3</sup>Galbot 
    <h3 align="center"><a href="https://pku-epic.github.io/MaskClustering/">Project Page</a> | <a href="https://arxiv.org/abs/2401.07745">Paper</a></h3>
    <h3 align="center">CVPR 2024</h3>
    </p>
</p>

<br/>

Given an RGB-D scan and a reconstructed point cloud, MaskClustering leverages **multi-view verificatio**n to merge 2D instance masks in each frame into 3D instances, achieving strong zero-shot open-vocabulary 3D instance segmentation performance on the ScanNet, ScanNet++, and MatterPort3D datasets.
![teaser](./figs/teaser.png)

# Fast Demo
Step 1: Install dependencies

First, install PyTorch following the [official instructions](https://pytorch.org/), e.g., for CUDA 11.8.:
```bash
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
Then, install Pytorch3D. You can try 'pip install pytorch3d', but it doesn't work for me. Therefore I install it from source:
```bash
cd third_party
git clone git@github.com:facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .
```
Finally, install other dependencies:
```bash
cd ../..
pip install -r requirements.txt
```

Step 2: Download demo data from [Google Drive](https://drive.google.com/file/d/1uwhJB0LKoc2meEkIz6ravYbscdB91dw7/view?usp=sharing) or from [Baidu Drive](https://pan.baidu.com/s/1jbodgv-nSmIRvKZJb1zLPg?pwd=szrt) (password: szrt). Then unzip the data to ./data and your directory should look like this: data/demo/scene0608_00, etc.

Step 3: Run the clustering demo and visualize the class-agnostic result using Pyviz3d:
```bash
bash demo.sh
```

# Quantitative Results
In this section, we provide a comprehensive guide on installing the full version of MaskClustering, data preparation, and conducting experiments on the ScnaNet, ScanNet++, and MatterPort3D datasets.

## Further installation
To run the full pipeline of MaskClustering, you need to install 2D instance segmentation tool [Cropformer](https://github.com/qqlu/Entity) and [Open CLIP](https://github.com/mlfoundations/open_clip).

### CropFormer
The official installation of Cropformer is composed of two steps: installing detectron2 and then Cropformer. For your convenience, I have combined the two steps into the following scripts. If you have any problems, please refer to the original [Cropformer](https://github.com/qqlu/Entity/blob/main/Entityv2/CropFormer/INSTALL.md) installation guide.
```bash
cd third_party
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ../
git clone git@github.com:qqlu/Entity.git
cp -r Entity/Entityv2/CropFormer detectron2/projects
cd detectron2/projects/CropFormer/entity_api/PythonAPI
make
cd ../..
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
pip install -U openmim
mim install mmcv
```
We add an additional script into cropformer to make it sequentialy process all sequences.
```bash
cd ../../../../../../../../
cp mask_predict.py third_party/detectron2/projects/CropFormer/demo_cropformer
```
Finally, download the [CropFormer checkpoint](https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg/tree/main/CropFormer_model/Entity_Segmentation/Mask2Former_hornet_3x) and modify the 'cropformer_path' variable in script.py.

### CLIP
Install the open clip library by 
```bash
pip install open_clip_torch
```
For the checkpoint, when you run the script, it will automatically download the checkpoint. However, if you want to download it manually, you can download it from [here](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main) and set the path when loading CLIP model using 'create_model_and_transforms' function.

## Data Preparation
### ScanNet
Please follow the official [ScanNet](http://www.scan-net.org/ScanNet/) guide to sign the agreement and send it to scannet@googlegroups.com. After receiving the response, you can download the data. You only need to download the ['.aggregation.json', '.sens', '.txt', '_vh_clean_2.0.010000.segs.json', '_vh_clean_2.ply', '_vh_clean_2.labels.ply'] files. Please also set the 'label_map' on to download the 'scannetv2-labels.combined.tsv' file.

After downloading the data, you can run the following script to prepare the data. Please change the 'raw_data_dir', 'target_data_dir', 'split_file_path', 'label_map_file' and 'gt_dir' variables before you run.
```bash 
cd preprocess/scannet
python process_val.py
python prepare_gt.py
```
After running the script, you will get the following directory structure:
```
data/scannet
  ├── processed
      ├── scene0011_00
          ├── pose                            <- folder with camera poses
          │      ├── 0.txt 
          │      ├── 10.txt 
          │      └── ...  
          ├── color                           <- folder with RGB images
          │      ├── 0.jpg  (or .png/.jpeg)
          │      ├── 10.jpg (or .png/.jpeg)
          │      └── ...  
          ├── depth                           <- folder with depth images
          │      ├── 0.png  (or .jpg/.jpeg)
          │      ├── 10.png (or .jpg/.jpeg)
          │      └── ...  
          ├── intrinsic                 
          │      └── intrinsic_depth.txt       <- camera intrinsics
          |      └── ...
          └── scene0011_00_vh_clean_2.ply      <- point cloud of the scene
  └── gt                                       <- folder with ground truth 3D instance masks
      ├── scene0011_00.txt
      └── ...
```

### ScanNet++
Please follow the official [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) guide to sign the agreement and download the data. **In order to help reproduce the results, we provide the configs we use to download and preprocess the scannet++ in preprocess/scannetpp.** Please modify the paths in these configs and paste them to the corresponding folders before running the script. Then clone the [ScanNet++ toolkit](https://github.com/scannetpp/scannetpp).

To extract the rgb and depth image, run the following script:
```bash
  python -m iphone.prepare_iphone_data iphone/configs/prepare_iphone_data.yml
  python -m common.render common/configs/render.yml
```

Since the original mesh is of super high resolution, we downsample it and generate the ground truth accordingly as the following:
```bash
  python -m semantic.prep.prepare_training_data semantic/configs/prepare_training_data.yml
  python -m semantic.prep.prepare_semantic_gt semantic/configs/prepare_semantic_gt.yml
```

After running the script, you will get the following directory structure:
```
data/scannetpp
  ├── data
      ├── 0d2ee665be
          ├── iphone                            
          |       ├── rgb
          │         ├── frame_000000.jpg 
          │         ├── frame_000001.jpg 
          │         └── ... 
          |       ├── render_depth 
          │         ├── frame_000000.png 
          │         ├── frame_000001.png 
          │         └── ... 
          |       └── ... 
          └── scans                        
      └── ...
  ├── gt 
  ├── metadata
  ├── pcld_0.25     <- downsampled point cloud of the scene
  └── splits
```

### MatterPort3D
Please follow the official [MatterPort3D](https://github.com/niessner/Matterport) guide to sign the agreement and download the data. We use a subset of its testing scenes to ensure Mask3D remains within memory constraints. The list of scenes we use can be found in splits/matterport3d.txt. Download only the following: ['undistorted_color_images', 'undistorted_depth_images', 'undistorted_camera_parameters', 'house_segmentations']. Upon download, unzip the files. Your directory structure should resemble (or you can modify the paths in 'preprocess/matterport3d/process.py' and 'dataset/matterport.py'):
```
data/matterport3d/scans
  ├── 2t7WUuJeko7
      ├── 2t7WUuJeko7
          ├── house_segmentations
          |         ├── 2t7WUuJeko7.ply
          |         └── ...
          ├── undistorted_camera_parameters
          |         └── 2t7WUuJeko7.conf
          ├── undistorted_color_images
          |         ├── xxx_i0_0.jpg
          |         └── ...
          └── undistorted_depth_images
                    ├── xxx_d0_0.png
                    └── ...
  ├── ARNzJeq3xxb
  ├── ...
  └── YVUC4YcDtcY
```
Then run the following script to prepare the ground truth:
```bash
cd preprocess/matterport3d
python process.py
```

## Running Experiments
Simply find the corresponding config in the 'configs' folder and run the following command. **Remember to change the 'cropformer_path' variable in the config and the 'CUDA_LIST' variable in the run.py.**
```bash
  python run.py --config config_name
```
For example, to run the ScanNet experiment, you can run the following command:
```bash
  python run.py --config scannet
```
This run.py will get the 2D instance masks, run mask clustering, get open-vocabulary features and evaluate the results. The evaluation results will be saved in the 'data/evaluation' folder.


### Time cost
We report the GPU hour of each step on Nvidia 3090 GPU.
|              | 2D mask prediction | mask clustering | CLIP feature extraction |   Overall  | time per scene  |
| :-: | :-: | :-: | :-: |:-:| :-:|
| ScanNet      |         5          |       6.5       |            2            |    13.5    |     2.6 min     |
| ScanNet++    |        4.5         |        4        |           0.5           |      9     |      10min      |
| MatterPort3D |        0.5         |        1        |           0.25          |      2     |     15 min      |



## Visualization
To visualize the 3D class-agnostic result of one specific scene, run the following command:
```bash
  python -m visualize.vis_scene --config scannet --seq_name scene0608_00
```