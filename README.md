# Mask2Hand

PyTorch implementation of "Mask2Hand: Learning to Predict the 3D Hand Pose and Shape from Shadow", \
Li-Jen Chang, Yu-Cheng Liao, Chia-Hui Lin, and Hwann-Tzong Chen \
*arXiv preprint arXiv:2205.15553* \
[**[Paper]**](https://arxiv.org/abs/2205.15553)

## Environment Setup
+ Create the environment from the provided `environment.yml` file.
  ```
  cd Mask2Hand
  conda env create -f environment.yml
  conda activate pytorch3d
  ```
+ Download the pretrained model from [Dropbox link](https://www.dropbox.com/s/mujjj8ov5e8r9ok/model_pretrained.pth?dl=1) and put it in the directory `checkpoint` using the following commands.
  ```
  mkdir -p ./checkpoint
  wget -O ./checkpoint/model_pretrained.pth https://www.dropbox.com/s/mujjj8ov5e8r9ok/model_pretrained.pth?dl=1
  ```
+ Download FreiHAND Dataset v2 from the [official website](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html) and unzip it into the directory `dataset/freihand/`.
  ```
  mkdir -p ./dataset/freihand
  cd dataset
  wget https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip
  unzip -q ./FreiHAND_pub_v2.zip -d ./freihand
  cd ..
  ```

## Run a Demo
+ If you want to use GPU, run
  ```
  CUDA_VISIBLE_DEVICES=0 python demo.py
  ```
+ Otherwise, run
  ```
  python demo.py
  ```
+ The demo results will be saved in the directory `demo_output`.

## Evaluation
+ Calculate the error of the predicted hand joints and meshes
  ```
  CUDA_VISIBLE_DEVICES=0 python test.py
  ```
+ Calculate the mIoU between the ground-truth and projected silhouettes
  ```
  CUDA_VISIBLE_DEVICES=0 python test_iou.py
  ```

## Training
Run the following script to train a model from scratch.
```
CUDA_VISIBLE_DEVICES=0 python train.py
```

## Citation
```
@article{chang2022mask2hand,
  author={Li-Jen Chang and Yu-Cheng Liao and Chia-Hui Lin and Hwann-Tzong Chen},
  title={Mask2Hand: Learning to Predict the 3D Hand Pose and Shape from Shadow},
  journal={CoRR},
  volume={abs/2205.15553},
  year={2022}
}
```

## Acknowledgement
The PyTorch implementation of MANO comes from [GrabNet](https://github.com/otaheri/MANO) and some visualization utilities are modified from [CMR](https://github.com/SeanChenxy/HandMesh).

