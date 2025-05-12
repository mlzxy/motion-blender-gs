#! /usr/bin/env python3
import torch
import time
from loguru import logger
import fire
import shutil
from pathlib import Path
import json
import subprocess as sp
import os.path as osp

import os
import numpy as np
import glob
import json
from PIL import Image
from tqdm import tqdm
import shutil
import sys
import numpy as np
import sqlite3
import open3d as o3d
from motionblender.lib.misc import load_json
from motionblender.lib.convert_utils import from_camera_json, to_camera_json
import sys

def process_ply_file(input_file, output_file, threshold=40000):
    # 读取输入的ply文件
    pcd = o3d.io.read_point_cloud(str(input_file))
    logger.info(f"Total points: {len(pcd.points)}")

    # 通过点云下采样将输入的点云减少
    voxel_size=0.02
    while len(pcd.points) > threshold:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        logger.info(f"Downsampled points: {len(pcd.points)}")
        voxel_size+=0.01

    # 将结果保存到输入的路径中
    o3d.io.write_point_cloud(str(output_file), pcd)


def bash(cmd, retry=1):
    logger.info(f"\033[91m{cmd}\033[0m")
    while retry > 0:
        retry -= 1
        x = sp.run(cmd, shell=True)
        if x.returncode == 0:
            return
        if retry > 0:
            time.sleep(10)
            
        
    if x.returncode != 0:
        raise ValueError(f"command \033[91m{cmd}\033[0m failed with return code {x.returncode}")




def array_to_blob(array):
    return array.tostring()

def blob_to_array(blob, dtype, shape=(-1,)):
    return np.fromstring(blob, dtype=dtype).reshape(*shape)

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=True WHERE camera_id=?",
            (model, width, height, array_to_blob(params),camera_id))
        return cursor.lastrowid

def camTodatabase(database_path, txt_path):
    camModelDict = {'SIMPLE_PINHOLE': 0,
                    'PINHOLE': 1,
                    'SIMPLE_RADIAL': 2,
                    'RADIAL': 3,
                    'OPENCV': 4,
                    'FULL_OPENCV': 5,
                    'SIMPLE_RADIAL_FISHEYE': 6,
                    'RADIAL_FISHEYE': 7,
                    'OPENCV_FISHEYE': 8,
                    'FOV': 9,
                    'THIN_PRISM_FISHEYE': 10}
    if os.path.exists(database_path)==False:
        logger.info("ERROR: database path dosen't exist -- please check database.db.")
        return
    # Open the database.
    db = COLMAPDatabase.connect(database_path)

    idList=list()
    modelList=list()
    widthList=list()
    heightList=list()
    paramsList=list()
    # Update real cameras from .txt
    with open(txt_path, "r") as cam:
        lines = cam.readlines()
        for i in range(0,len(lines),1):
            if lines[i][0]!='#':
                strLists = lines[i].split()
                cameraId=int(strLists[0])
                cameraModel=int(camModelDict[strLists[1]]) #SelectCameraModel
                width=int(float(strLists[2]))
                height=int(float(strLists[3]))
                paramstr=np.array(strLists[4:12])
                params = paramstr.astype(np.float64)
                idList.append(cameraId)
                modelList.append(cameraModel)
                widthList.append(width)
                heightList.append(height)
                paramsList.append(params)
                camera_id = db.update_camera(cameraModel, width, height, params, cameraId)

    # Commit the data to the file.
    db.commit()
    # Read and check cameras.
    rows = db.execute("SELECT * FROM cameras")
    for i in range(0,len(idList),1):
        camera_id, model, width, height, params, prior = next(rows)
        params = blob_to_array(params, np.float64)
        assert camera_id == idList[i], f"{camera_id} != {idList[i]}"
        assert model == modelList[i] and width == widthList[i] and height == heightList[i]
        assert np.allclose(params, paramsList[i])

    # Close database.db.
    db.close()



def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def to_colmap_input_fmt(root_dir, factor=1, every=1, scene_norm_dict=None):

    colmap_dir = os.path.join(root_dir,"sparse_")
    if not os.path.exists(colmap_dir):
        os.makedirs(colmap_dir)
    imagecolmap_dir = os.path.join(root_dir,"image_colmap")
    if not os.path.exists(imagecolmap_dir):
        os.makedirs(imagecolmap_dir)

    image_dir = os.path.join(root_dir,"rgb",f"{factor}x")
    images = os.listdir(image_dir)
    images.sort()
    camera_dir = os.path.join(root_dir,"camera")
    cameras = os.listdir(camera_dir)
    cameras.sort()
    cams = []
    for jsonfile in tqdm(cameras):
        if scene_norm_dict is not None:
            K, c2w, img_wh = from_camera_json(os.path.join(camera_dir,jsonfile))
            w2c = torch.linalg.inv(c2w)
            w2c = w2c @ torch.linalg.inv(scene_norm_dict['transfm'])
            w2c[:3, 3] /= scene_norm_dict['scale']
            cams.append(to_camera_json(K, torch.linalg.inv(w2c), *img_wh))
        else:
            with open(os.path.join(camera_dir,jsonfile)) as f:
                cams.append(json.load(f))
    image_size = cams[0]['image_size']
    image = Image.open(os.path.join(image_dir,images[0]))
    size = image.size
    object_images_file = open(os.path.join(colmap_dir,"images.txt"),"w")
    object_cameras_file = open(os.path.join(colmap_dir,"cameras.txt"),"w")

    idx=0
    cnt=0
    sizes=1
    while len(cams)//sizes > 200:
        sizes += 1
    for cam, image in zip(cams, images):
        cnt+=1

        if cnt % every != 0:
            continue

        R = np.array(cam['orientation']).T
        T = -np.array(cam['position'])@R

        T = [str(i) for i in T]
        qevc = [str(i) for i in rotmat2qvec(R.T)]
        print(idx+1," ".join(qevc)," ".join(T),1,image,"\n",file=object_images_file)
        print(idx+1,"SIMPLE_PINHOLE",image_size[0]/factor,image_size[1]/factor,cam['focal_length']/factor,cam['principal_point'][0]/factor,cam['principal_point'][1]/factor,file=object_cameras_file)
        idx+=1
        shutil.copy(os.path.join(image_dir,image),os.path.join(imagecolmap_dir,image))
    print(idx)
    # write camera infomation.
    object_point_file = open(os.path.join(colmap_dir,"points3D.txt"),"w")

    object_cameras_file.close()
    object_images_file.close()
    object_point_file.close()

local_root = Path(os.environ['DATA_ROOT'])


def find_factor(factor, workdir, scene):
    if factor == 'smallest':
        factor = min([int(s[:-1]) for s in os.listdir(str(workdir / 'rgb'))])
        logger.info(f"Using smallest factor: {factor} for {scene}")
    return factor


def prepare(scene, factor='smallest', every=1, normalize_scene=False, **kwargs):
    workdir = local_root / scene
    if normalize_scene:
        logger.warning("normalize_scene is enabled!")
        scene_norm_dict = torch.load(workdir / "cache" /  "scene_norm_dict.pth")
        logger.info(f"scene_norm_dict: {scene_norm_dict}")
    else:
        scene_norm_dict = None

    factor = find_factor(factor, workdir, scene)
    logger.info(f"Preparing {scene} with factor {factor}")
    time.sleep(1)


    shutil.rmtree(workdir / 'sparse_', ignore_errors=True)
    shutil.rmtree(workdir / 'image_colmap', ignore_errors=True)
    to_colmap_input_fmt(workdir, factor, every, scene_norm_dict)
    shutil.rmtree(workdir / 'colmap', ignore_errors=True)
    shutil.rmtree(workdir / 'colmap/sparse/0', ignore_errors=True)
    os.makedirs(workdir / 'colmap', exist_ok=True)
    shutil.move(workdir / 'image_colmap', workdir / 'colmap/images')
    shutil.move(workdir / 'sparse_', workdir / 'colmap/sparse_custom')


def run(scene, **kwargs):
    workdir = local_root / scene
    # 16384 previous max num of features
    bash(f'colmap feature_extractor --database_path {workdir / "colmap/database.db"} --image_path {workdir / "colmap/images"} --SiftExtraction.max_image_size 4096 --SiftExtraction.max_num_features 65536 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1')
    camTodatabase(workdir / 'colmap/database.db', workdir / 'colmap/sparse_custom/cameras.txt')
    bash(f'colmap exhaustive_matcher --database_path {workdir / "colmap/database.db"}', retry=10)
    os.makedirs(workdir / 'colmap/sparse/0', exist_ok=True)
    bash(f'colmap point_triangulator --database_path {workdir / "colmap/database.db"} --image_path {workdir / "colmap/images"} --input_path {workdir / "colmap/sparse_custom"} --output_path {workdir / "colmap/sparse/0"} --clear_points 1')
    os.makedirs(workdir / 'colmap/dense/workspace', exist_ok=True)
    bash(f'colmap image_undistorter --image_path {workdir / "colmap/images"} --input_path {workdir / "colmap/sparse/0"} --output_path {workdir / "colmap/dense/workspace"}')
    bash(f'colmap patch_match_stereo --workspace_path {workdir / "colmap/dense/workspace"}')
    bash(f'colmap stereo_fusion --workspace_path {workdir / "colmap/dense/workspace"} --output_path {workdir / "colmap/dense/workspace/fused.ply"}')
    process_ply_file(workdir / 'colmap/dense/workspace/fused.ply', workdir / 'points3D_downsample2.ply')


def clean_dense_dir(scene):
    workdir = local_root / scene
    shutil.rmtree(workdir / 'colmap' / 'dense', ignore_errors=True)

def prepare_and_run(scene, **kwargs):
    prepare(scene, **kwargs)
    run(scene, **kwargs)


def extract_pcd_each_instance(scene, factor=2, image_ext='.png'):
    logger.info("this script assumes that the whole scene is already reconstructed by colmap.")

    workdir = local_root / scene
    factor = find_factor(factor, workdir, scene)
    inp_maskdir = workdir / 'instance' / f'{factor}x' / 'imask'
    mask_workdir = workdir / 'colmap' / 'instance'
    shutil.rmtree(mask_workdir, ignore_errors=True)
    inst_names = [f'{name.split(":")[0]}-{name_i+1}' for name_i, name in enumerate(json.load(open(workdir / f'instance/{factor}x/names.json')))]
    inst_names = ['bg'] + inst_names
    masks = [(np.array(Image.open(inp_maskdir / f)), f) for f in sorted(os.listdir(inp_maskdir)) if f.endswith('.png')]
    for inst_i, inst_name in enumerate(inst_names):
        target_dir = mask_workdir / inst_name / 'masks' # only duplicate masks
        target_dir.mkdir(parents=True, exist_ok=True)
        for mask, name in tqdm(masks, desc='saving masks'):
            fig = Image.fromarray(mask == inst_i)
            fig.save(target_dir / (name.replace('.png', image_ext) + '.png'))

    for inst_name in inst_names:
        target_dir = mask_workdir / inst_name
        bash(f'colmap stereo_fusion --workspace_path {workdir / "colmap/dense/workspace"}  --StereoFusion.mask_path {target_dir / "masks"} --output_path {target_dir / "fused.ply"}')
        process_ply_file(target_dir / 'fused.ply', workdir / f'pcd.{inst_name}.ply', threshold=100_000 if inst_name == 'bg' else 40_000)

if __name__ == "__main__":
    fire.Fire({
        'prepare': prepare,
        'run': run,
        'prepare-and-run': prepare_and_run,
        'clean_dense_dir': clean_dense_dir,
        'extract_pcd_each_instance': extract_pcd_each_instance
    })
