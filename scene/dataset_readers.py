#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import cv2
import polanalyser as pa
from utils.stokes_utils import *

def compute_dop_aop(stokes_world):
    s0 = stokes_world[:,:,0]
    s1 = stokes_world[:,:,1]
    s2= stokes_world[:,:,2]
    DoP = np.sqrt(s1**2 + s2**2) / (s0 + 1e-8)
    DoP = np.clip(DoP, 0, 1)
    AoP = 0.5 * np.arctan2(s2, s1)  # Angle in radians
    # AoP = np.degrees(AoP)  # Convert to degrees
    AoP = (AoP + np.pi) % np.pi
    return DoP, AoP

def decompose_raw_polar(path):


    image_array = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if image_array is None:
        raise RuntimeError(f"Failed to load image from {path}")
    I90  = cv2.cvtColor(image_array[0::2, 0::2], cv2.COLOR_BAYER_BG2GRAY)/255   # top-left
    I45  = cv2.cvtColor(image_array[0::2, 1::2], cv2.COLOR_BAYER_BG2GRAY)/255  # top-right
    I135 = cv2.cvtColor(image_array[1::2, 0::2], cv2.COLOR_BAYER_BG2GRAY) /255  # bottom-left
    I0   = cv2.cvtColor(image_array[1::2, 1::2], cv2.COLOR_BAYER_BG2GRAY) /255  # bottom-right

    return I0, I45, I90, I135

def compute_local_stokes_rgb(I0_bgr, I45_bgr, I90_bgr, I135_bgr):
        angles = np.deg2rad([0, 45, 90,135])
        I0   = I0_bgr.astype(np.float32) / 255.
        I45  = I45_bgr.astype(np.float32) / 255.
        I90 = I90_bgr.astype(np.float32) / 255.
        I135 = I135_bgr.astype(np.float32) / 255.
         # Compute Stokes vectors separately for each RGB channel
        img_stokes_R = pa.calcStokes([I0[..., 0], I45[..., 0], I90[..., 0], I135[..., 0]], angles)
        img_stokes_G = pa.calcStokes([I0[..., 1], I45[..., 1], I90[..., 1], I135[..., 1]], angles)
        img_stokes_B = pa.calcStokes([I0[..., 2], I45[..., 2], I90[..., 2], I135[..., 2]], angles)

        # Merge the separate per-channel Stokes parameters into an RGB image
        s0_merged = np.stack([img_stokes_R[..., 0], img_stokes_G[..., 0], img_stokes_B[..., 0]], axis=-1)
        s1_merged = np.stack([img_stokes_R[..., 1], img_stokes_G[..., 1], img_stokes_B[..., 1]], axis=-1)
        s2_merged = np.stack([img_stokes_R[..., 2], img_stokes_G[..., 2], img_stokes_B[..., 2]], axis=-1)

        stokes_local = np.stack([s0_merged,s1_merged,s2_merged], axis=-1)
        return stokes_local

def compute_local_stokes_gray(I0_gray, I45_gray, I90_gray, I135_gray):
        s0 = 0.5*(I0_gray+ I45_gray+ I90_gray+ I135_gray)
        s1 = I0_gray - I90_gray
        s2 = I45_gray - I135_gray
        s3 = np.zeros_like(s0, dtype=np.float32)
    
        s0 = np.clip(s0, 0, 1)
        s1 = np.clip(s1, -1, 1)
        s2 = np.clip(s2, -1, 1)
 
        stokes_local = np.stack([s0, s1, s2], axis=-1).astype(np.float32)
        
        # angles = np.deg2rad([0, 45, 90,135])
        # stokes_local = pa.calcStokes([I0_gray, I45_gray, I90_gray, I135_gray], angles)
        # stokes_local[:,:,0] = np.clip(stokes_local[:,:,0], 0, 1)
        # stokes_local[:,:,1] = np.clip(stokes_local[:,:,0], -1, 1)
        # stokes_local[:,:,2] = np.clip(stokes_local[:,:,0], -1, 1)
        return stokes_local
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    # semantic_image: np.array
    # semantic_path: str
    # semantic_image_name: str
    # semantic_classes: list
    # num_semantic_classes: int
    semantic_image: None
    semantic_path: None
    semantic_image_name: None
    semantic_classes: None
    num_semantic_classes: None
    width: int
    height: int
    stokes_world: np.array = None
    dop : np.array = None
    aop: np.array = None
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        I0_gray, I45_gray, I90_gray, I135_gray = decompose_raw_polar(image_path)
        avg_rgb = 0.25*(I0_gray.astype(np.float32) + I0_gray.astype(np.float32) +
                        I0_gray.astype(np.float32) + I0_gray.astype(np.float32))
        avg_rgb = avg_rgb.clip(0,255).astype(np.uint8)
        image = Image.fromarray(avg_rgb)

  



      
        #rotate stokes vector
        stokes_local = compute_local_stokes_gray(I0_gray, I45_gray, I90_gray, I135_gray)
    
      
        #rotate stokes vector
        cam2world = R # rotation matrix.
        x_cam, y_cam, v_cam = stokes_basis_cam(pixel_H=height, pixel_W=width)

        x_world, x_world_target = stokes_basis_cam_to_world(cam2world, v_cam, x_cam)

        rot_mat = rotate_mat_stokes_basis(v_cam, x_world, x_world_target)

        stokes_target = rotate_stokes_vector(rot_mat, stokes_local)
        stokes_world = stokes_target.transpose(1,2,0)
       
        img_dolp,img_aolp = compute_dop_aop(np.array(stokes_world))
        # print(img_dolp.min(), img_dolp.max())
        # print(img_aolp.min(), img_aolp.max())
        # img_dolp = pa.cvtStokesToDoLP(stokes_world)
        # img_aolp = pa.cvtStokesToAoLP(stokes_world)
  

        #Inv Rotation Check->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # print(stokes_world.shape)
        
        # rot_mat_inv = torch.linalg.inv(rot_mat)
        
        # stokes_pred_local = rotate_stokes_vector(rot_mat_inv, stokes_world)
        # stokes_local_rot   = stokes_pred_local
        # print(">>>>>>>>>>>>>>>>>>>>>>>>")
        
     
        # print(stokes_local_rot.shape)
        # print(stokes_local_rot[:,:,0].shape)
        # import matplotlib.pyplot as plt
        # plt.imshow(img_aolp,cmap="hsv")
        # plt.title("Averaged RGB Image from Polarized Inputs")
        # plt.axis('off')
        # plt.show()
        # input('q')
        # assert torch.allclose(
        #     torch.tensor(stokes_local_rot, dtype=torch.float32), 
        #     torch.tensor(stokes_local, dtype=torch.float32), 
        #     atol=1e-3
        # )
        # input('q')
        #Inv Rotation Check->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # semantic_image
        base_dir, _ = os.path.split(images_folder)
        # semantic_folder = os.path.join(base_dir, "semantic_class")
        # semantic_path = os.path.join(semantic_folder, os.path.basename(extr.name))

        # prefix = '0000_'
        # semantic_image_name = prefix + extr.name

        # semantic_path = os.path.join(semantic_folder, os.path.basename(semantic_image_name))
        # semantic_image = np.array(Image.open(semantic_path))

        # semantic_classes
        # semantic_all_imgs = np.empty(0)
        # semantic_all_files = [file for file in os.listdir(semantic_folder) if file.endswith(".png")]
        # for semantic_file in semantic_all_files:
        #     semantic_img_path = os.path.join(semantic_folder, semantic_file)
        #     semantic_img = np.array(Image.open(semantic_img_path))
        #     if semantic_all_imgs.size == 0:
        #         semantic_all_imgs = semantic_img
        #     else:
        #         semantic_all_imgs = np.concatenate((semantic_all_imgs, semantic_img),axis=0)
        #
        # semantic_classes = np.unique(semantic_all_imgs).astype(np.uint8)
        # num_semantic_classes = semantic_classes.shape[0]
        # semantic_classes = list(semantic_classes)

        # 快速
        # num_semantic_classes = 16
        # semantic_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name,
                              semantic_image=None, semantic_path=None,
                              semantic_image_name=None,
                              semantic_classes=None, num_semantic_classes=None,
                              width=width, height=height,stokes_world=stokes_world,dop = img_dolp, aop= img_aolp)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]

        # train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx % 2 == 0]
        # test_cam_infos = [c for idx, c in enumerate(test_cam_infos) if idx % 4 == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}