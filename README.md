<p align="center">

  <h1 align="center">MaskClustering</h1>
    <p align="center">
        <a href="https://miyandoris.github.io/">Mi Yan</a><sup>1,2</sup></span>, 
        <a href="https://jzhzhang.github.io/">Jiazhao Zhang</a><sup>1,2</sup>
        <a href="https://github.com/fzy139/">Yan Zhu</a><sup>1</sup>, 
        <a href="https://hughw19.github.io/">He Wang</a><sup>1,2,3</sup>, 
        <br>
        <sup>1</sup>Peking University, 
        <sup>2</sup>Beijing Academy of Artificial Intelligence, 
        <sup>3</sup>Galbot 
    <h3 align="center"><a href="https://pku-epic.github.io/MaskClustering/">Project Page</a> | <a href="https://arxiv.org/abs/2401.07745">Paper</a></h3>
    <h2 align="center">CVPR 2024</h2>
    </p>
</p>

<br/>

????
![teaser](./figs/teaser.png)

## Fast demo
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

Step 2: Download demo data from [Google Drive](https://drive.google.com/file/d/1uwhJB0LKoc2meEkIz6ravYbscdB91dw7/view?usp=sharing) or from [Baidu Drive](https://pan.baidu.com/s/1jbodgv-nSmIRvKZJb1zLPg?pwd=szrt) (code: szrt). Then unzip the data to ./data and your directory should look like this: data/demo/scene0608_00, etc.

Step 3: Run the clustering demo and visualize the class-agnostic result using Pyviz3d:
```bash
bash demo.sh
```

?????