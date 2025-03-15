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

import torch
import numpy as np
from scene import Scene
import time
import os
import polanalyser as pa
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import imageio
from utils.general_utils import safe_state, calculate_reflection_direction, visualize_normal_map
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr, calculate_segmentation_metrics, evaluate,psnr_np
from mynet import MyNet, embedding_fn
from utils.graphics_utils import views_dir
import time
from utils.stokes_utils import *
import cv2

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
def compute_stokes_rmse(stokes_gt, stokes_pred):

    rmse_values = {}
    for i, label in enumerate(["S0", "S1", "S2"]):
        rmse = torch.sqrt(torch.mean((stokes_pred[:, :, i] - stokes_gt[:, :, i]) ** 2))
        rmse_values[label] = rmse
    return rmse_values

def compute_stokes_psnr(stokes_gt, stokes_pred):
    psnr_values = {}
    for i, label in enumerate(["S0", "S1", "S2"]):
        psnr_values[label] = psnr(torch.tensor(stokes_pred[:, :, i]),torch.tensor(stokes_gt[:, :, i])).mean().double()
    return psnr_values

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    net_path = os.path.join(model_path, "model_ckpt{}pth".format(iteration))
    model = MyNet().to("cuda")
    net_weights = torch.load(net_path)
    model.load_state_dict(net_weights)

    model.eval()

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    stokes_path = os.path.join(model_path, name, "ours_{}".format(iteration), "stokes_visualization")
    mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mask")
    f_path = os.path.join(model_path, name, "ours_{}".format(iteration), "f")
    highlight_path = os.path.join(model_path, name, "ours_{}".format(iteration), "highlight")
    color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "color")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal")

    # makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)
    # makedirs(stokes_path, exist_ok=True)
    # makedirs(mask_path, exist_ok=True)
    # makedirs(f_path, exist_ok=True)
    # makedirs(highlight_path, exist_ok=True)
    makedirs(color_path, exist_ok=True)
    # makedirs(normal_path, exist_ok=True)

    all_time = []
    all_psnr_values = {"S0": [], "S1": [], "S2": []}
    all_rmse_values = {"S0": [], "S1": [], "S2": []}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        start_time = time.time()

        render_pkg = render(view, gaussians, pipeline, background)

        rendered_features, viewspace_point_tensor, visibility_filter, radii, depths, _ = render_pkg

        views_emd = view.views_emd

        rendered_features[0] = torch.cat((rendered_features[0], views_emd), dim=1)
        image = model(*rendered_features)
        # rendering = image['im_out'].squeeze(0)

        all_time.append(time.time() - start_time)

        gt = view.original_image[0:3, :, :]
        stokes_gt =torch.from_numpy(view.stokes_world).to('cuda') 
      
        stokes_pred_world = image['s_out'].squeeze(0) 
        stokes_pred_world = stokes_pred_world.permute(1, 2, 0)
        stokes_pred_world_np = stokes_pred_world.detach().cpu().numpy()
        img_dolp,img_aolp=compute_dop_aop(stokes_pred_world_np)
        # img_dolp = pa.cvtStokesToDoLP(stokes_pred_world_np)
        # img_aolp = pa.cvtStokesToAoLP(stokes_pred_world_np)
        # img_intensity = pa.cvtStokesToIntensity(stokes_pred_world_np)
        # import matplotlib.pyplot as plt
        # plt.imshow(img_dolp,cmap="viridis")
        # plt.title("Averaged RGB Image from Polarized Inputs")
        # plt.axis('off')
        # plt.show()
        # input('q')
        img_intensity = np.clip(stokes_pred_world_np[:,:,0], 0, 1)
        img_intensity_vis = np.clip(255 * ((img_intensity / 255 / 2) ** (1 / 2.2)), 0, 255).astype(np.uint8)
        img_dolp = np.clip(img_dolp, 0, 1)
        img_aolp_colormap = img_aolp / 180.0
        psnr_values = compute_stokes_psnr(stokes_gt, stokes_pred_world)
        rmse_values = compute_stokes_rmse(stokes_gt, stokes_pred_world)
        # Store PSNR values
        for key in all_psnr_values:
            all_psnr_values[key].append(psnr_values[key])

        for key in all_rmse_values:
            all_rmse_values[key].append(rmse_values[key])
            
        # # Save Visualizations
        # polar_vis(stokes_local_rot, stokes_path,idx)
        # input('q')
        # predicted_normals = image['normals'].squeeze(0)
        # save_normal_map(predicted_normals, os.path.join(normal_path, '{0:05d}.png'.format(idx)))
       
        
        # torchvision.utils.save_image(torch.tensor(stokes_pred_local[:,:,0]), os.path.join(stokes_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        #
        # torchvision.utils.save_image(image['mask_out'].squeeze(0),
        #                              os.path.join(mask_path, '{0:05d}'.format(idx) + ".png"))
        import matplotlib.pyplot as plt
        plt.imshow(img_dolp,cmap="gray")
        plt.title("Averaged RGB Image from Polarized Inputs")
        plt.axis('off')
        plt.show()
        input('q')
        cv2.imwrite(os.path.join(color_path, '{0:05d}'.format(idx) + "_AoP.png"), pa.applyColorToAoLP(img_aolp_colormap))
        cv2.imwrite(os.path.join(color_path, '{0:05d}'.format(idx) + "_DoP.png"),pa.applyColorToDoP(img_dolp))
        cv2.imwrite(os.path.join(color_path, '{0:05d}'.format(idx) + "_intensity.png"),img_intensity)
        # torchvision.utils.save_image(torch.tensor(pa.applyColorToDoP(img_dolp)),
        #                              os.path.join(color_path, '{0:05d}'.format(idx) + "_DoP.png"))
        # torchvision.utils.save_image(torch.tensor(img_intensity),
        #                              os.path.join(color_path, '{0:05d}'.format(idx) + "_intensity.png"))
    

    # Compute average PSNR
    # Compute average PSNR for each Stokes component
    avg_psnr_values = {key: sum(values) / len(values) for key, values in all_psnr_values.items()}

    # Print PSNR values
    print("\n📌 Average PSNR for Each Stokes Component:")
    for key, value in avg_psnr_values.items():
        print(f"  {key}: {value:.2f} dB")

    # Compute total average PSNR across all components
    avg_psnr_total = sum(avg_psnr_values.values()) / 4
    print(f"\n✅ Overall Average Stokes PSNR: {avg_psnr_total:.2f} dB")
    print("Average time per image: {}".format(sum(all_time) / len(all_time)))
    print("Render FPS: {}".format(1 / (sum(all_time) / len(all_time))))

    # Compute average RMSE for each Stokes component
    avg_rmse_values = {key: sum(values) / len(values) for key, values in all_rmse_values.items()}

    # Print RMSE values
    print("\n📌 Average RMSE for Each Stokes Component:")
    for key, value in avg_rmse_values.items():
        print(f"  {key}: {value:.4f}")

    # Compute total average RMSE across all components
    avg_rmse_total = sum(avg_rmse_values.values()) / 4
    print(f"\n✅ Overall Average Stokes RMSE: {avg_rmse_total:.4f}")

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.num_sem_classes)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # for camera in scene.getTestCameras():
        #     print(camera.R)

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline,
                       background)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline,
                       background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)