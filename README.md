# MaskClustering: View Consensus based Mask Graph Clustering for Open-Vocabulary 3D Instance Segmentation

### [Project Page](https://pku-epic.github.io/MaskClustering/) | [Paper](https://arxiv.org/abs/2401.07745)

[Mi Yan](https://miyandoris.github.io/), [Jiazhao Zhang](https://jzhzhang.github.io/), [Yan Zhu](https://github.com/fzy139/), [He Wang](https://hughw19.github.io/)
<br/>

![teaser](./demo/teaser.png)


## Inference
### install mask2former
1. download the code
```
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
cd ../../../../../../../
cp generate_mask.py detectron2/projects/CropFormer/demo_cropformer
cd ../../../../../../../
cp mask_predict.py detectron2/projects/CropFormer/demo_cropformer
rm -r Entity
```

2. download the model from https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg/tree/main/CropFormer_model/Entity_Segmentation/Mask2Former_hornet_3x