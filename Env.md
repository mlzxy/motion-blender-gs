# Environment Setup


Please follow the script below, assuming that you have CUDA 12.x installed in your Linux machine. The code is tested against Python 3.10 + Ubuntu 22.04. As in the following, we provide snapshots of our environments, but these snapshots may only work in a x64 Py3.10 Linux machine.



```bash
conda create mbgs python=3.10
conda activate mbgs
pip install -r ./requirements.txt

# install gsplat
wget https://github.com/nerfstudio-project/gsplat/releases/download/v1.4.0/gsplat-1.4.0%2Bpt24cu121-cp310-cp310-linux_x86_64.whl
pip install ./gsplat-1.4.0+pt24cu121-cp310-cp310-linux_x86_64.whl

# install pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git" -U  -v --force-reinstall
```


We also provide a snapshot of our conda environment, this snapshot can be downloaded from `env/mbgs-env.tar` at [https://drive.google.com/drive/folders/1soX0ivDm7GIIVxs1fX6OXIDWEWrLzM6a?usp=sharing](https://drive.google.com/drive/folders/1soX0ivDm7GIIVxs1fX6OXIDWEWrLzM6a?usp=sharing). You can download it and extract it, and then run 

```bash
conda activate ./mbgs-env
```

To directly use this snapshoted environment, or verify python package versions against it. 



## Tools' Environment

> Optional, only required if you want to run the preprocessing tools.

You can setup and download checkpoints from their repos: 

- [https://github.com/IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)
- [https://github.com/facebookresearch/sapiens](https://github.com/facebookresearch/sapiens) Only needs to configure the pose estimation (keypoints) part. 


and download `weights/other/bootstapir_checkpoint_v2.pt` (from the google drive above) into `outputs/weights/other/bootstapir_checkpoint_v2.pt`.  This model is for the 2D tracking data generation from shape-of-motion.


To get started easily, you can also download our snapshotted `Grounded-SAM-2.tar.gz` and `sapiens.tar.gz` from the `envs` folder of our google drive, and extract them in the root folder of this repo. Note that even if you use the snapshotted folder, you still need to go through the requirement installation for `sapiens` (e.g., mmdet).


---

You are also recommended to install [torch-batch-svd](https://github.com/KinglittleQ/torch-batch-svd) if you enable the as-rigid-as-possible regularization (through `--loss.w-arap 0.1` for example). But this regularization is only used for the robot cloth scene.