import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trimesh
import cv2
from PIL import Image
import random
import json
import argparse

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import FreiHandDataset
from model import HandSilhouetteNet3
from eval_utils import EvalUtil, align_w_scale


def evaluate_model(model, dataloader, test_vertices = True, device = 'cpu'):
    # Initialize evaluation utilities
    eval_xyz, eval_xyz_aligned = EvalUtil(), EvalUtil()
    eval_mesh_err, eval_mesh_err_aligned = EvalUtil(num_kp = 778), EvalUtil(num_kp = 778)
    
    eval_xyz_refined, eval_xyz_refined_aligned = EvalUtil(), EvalUtil()
    eval_mesh_err_refined, eval_mesh_err_refined_aligned = EvalUtil(num_kp = 778), EvalUtil(num_kp = 778)

    # Evaluation phase
    model.eval()
    with torch.no_grad():
        # Iterate over the dataset once
        for i, (inputs, focal_lens, image_refs, joints_anno, dist_maps, meshes_anno) in enumerate(dataloader):
            inputs = inputs.to(device)
            focal_lens = focal_lens.to(device)
            image_refs = image_refs.to(device)
            dist_maps = dist_maps.to(device)
            joints_anno = joints_anno.to(device)
            if test_vertices:
                meshes_anno = meshes_anno.to(device)

            outputs = model(inputs, focal_lens, image_refs)
            
            # Unaligned joint position errors
            eval_xyz.feed(keypoint_gt = joints_anno, keypoint_pred = outputs['joints'])

            eval_xyz_refined.feed(keypoint_gt = joints_anno, keypoint_pred = outputs['refined_joints'])

            # Aligned joint position errors
            xyz_pred_aligned = []
            for i in range(joints_anno.shape[0]):
                xyz_pred_aligned.append(align_w_scale(joints_anno[i].cpu().numpy(), outputs['joints'][i].cpu().numpy()))
            xyz_pred_aligned = torch.tensor(xyz_pred_aligned)
            eval_xyz_aligned.feed(keypoint_gt = joints_anno.cpu(), keypoint_pred = xyz_pred_aligned)

            xyz_pred_refined_aligned = []
            for i in range(joints_anno.shape[0]):
                xyz_pred_refined_aligned.append(align_w_scale(joints_anno[i].cpu().numpy(), outputs['refined_joints'][i].cpu().numpy()))
            xyz_pred_refined_aligned = torch.tensor(xyz_pred_refined_aligned)
            eval_xyz_refined_aligned.feed(keypoint_gt = joints_anno.cpu(), keypoint_pred = xyz_pred_refined_aligned)

            if test_vertices:
                # Unaligned mesh vertex position errors
                eval_mesh_err.feed(keypoint_gt = meshes_anno, keypoint_pred = outputs['vertices'])

                eval_mesh_err_refined.feed(keypoint_gt = meshes_anno, keypoint_pred = outputs['refined_vertices'])

                # Aligned mesh vertex position errors
                verts_pred_aligned = []
                for i in range(meshes_anno.shape[0]):
                    verts_pred_aligned.append(align_w_scale(meshes_anno[i].cpu().numpy(), outputs['vertices'][i].cpu().numpy()))
                verts_pred_aligned = torch.tensor(verts_pred_aligned)
                eval_mesh_err_aligned.feed(keypoint_gt = meshes_anno.cpu(), keypoint_pred = verts_pred_aligned)

                verts_pred_refined_aligned = []
                for i in range(meshes_anno.shape[0]):
                    verts_pred_refined_aligned.append(align_w_scale(meshes_anno[i].cpu().numpy(), outputs['refined_vertices'][i].cpu().numpy()))
                verts_pred_refined_aligned = torch.tensor(verts_pred_refined_aligned)
                eval_mesh_err_refined_aligned.feed(keypoint_gt = meshes_anno.cpu(), keypoint_pred = verts_pred_refined_aligned)


        # Calculate results
        xyz_mean3d, _, xyz_auc3d, pck_xyz, thresh_xyz = eval_xyz.get_measures(0., 50., 100)     # using 100 equally spaced thresholds between 0 to 50 mm
        xyz_al_mean3d, _, xyz_al_auc3d, pck_xyz_al, thresh_xyz_al = eval_xyz_aligned.get_measures(0., 50., 100)

        xyz_refined_mean3d, _, xyz_refined_auc3d, _, _ = eval_xyz_refined.get_measures(0., 50., 100)     # using 100 equally spaced thresholds between 0 to 50 mm
        xyz_refined_al_mean3d, _, xyz_refined_al_auc3d, _, _ = eval_xyz_refined_aligned.get_measures(0., 50., 100)
        
        if test_vertices:
            mesh_mean3d, _, mesh_auc3d, pck_mesh, thresh_mesh = eval_mesh_err.get_measures(0., 50., 100)
            mesh_al_mean3d, _, mesh_al_auc3d, pck_mesh_al, thresh_mesh_al = eval_mesh_err_aligned.get_measures(0., 50., 100)

            mesh_refined_mean3d, _, mesh_refined_auc3d, _, _ = eval_mesh_err_refined.get_measures(0., 50., 100)
            mesh_refined_al_mean3d, _, mesh_refined_al_auc3d, _, _ = eval_mesh_err_refined_aligned.get_measures(0., 50., 100)

        # Output results
        print('[Evaluation of 3D Keypoints]')
        print(f'AUC of PCK = {xyz_auc3d}')
        print(f'MPJPE = {xyz_mean3d} mm\n')

        print('[Evaluation of Aligned 3D Keypoints]')
        print(f'AUC of PCK = {xyz_al_auc3d}')
        print(f'MPJPE = {xyz_al_mean3d} mm\n')

        print('[Evaluation of Refined 3D Keypoints]')
        print(f'AUC of PCK = {xyz_refined_auc3d}')
        print(f'MPJPE = {xyz_refined_mean3d} mm\n')

        print('[Evaluation of Aligned Refined 3D Keypoints]')
        print(f'AUC of PCK = {xyz_refined_al_auc3d}')
        print(f'MPJPE = {xyz_refined_al_mean3d} mm\n')

        if test_vertices:
            print('[Evaluation of 3D Mesh Vertices]')
            print(f'AUC of Percentage of Correct Vertices = {mesh_auc3d}')
            print(f'MPVPE = {mesh_mean3d} mm\n')

            print('[Evaluation of Aligned 3D Mesh Vertices]')
            print(f'AUC of Percentage of Correct Vertices = {mesh_al_auc3d}')
            print(f'MPVPE = {mesh_al_mean3d} mm\n')

            print('[Evaluation of Refined 3D Mesh Vertices]')
            print(f'AUC of Percentage of Correct Vertices = {mesh_refined_auc3d}')
            print(f'MPVPE = {mesh_refined_mean3d} mm\n')

            print('[Evaluation of Aligned Refined 3D Mesh Vertices]')
            print(f'AUC of Percentage of Correct Vertices = {mesh_refined_al_auc3d}')
            print(f'MPVPE = {mesh_refined_al_mean3d} mm\n')


def test(args):
    # Configurations
    joints_anno_file = 'training_xyz.json'
    camera_Ks_file = 'training_K.json'
    data_split_file = 'FreiHand_split_ids.json'
    vertices_anno_file = 'training_verts.json'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Data Loaders
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.8705], std = [0.3358])
    ])

    dataset_frei_test = FreiHandDataset(args.data_path, joints_anno_file, camera_Ks_file, data_split_file, vertices_anno_file, split = 'test', transform = transform, augment = False)
    dataloader_frei_test = DataLoader(dataset = dataset_frei_test, batch_size = args.batch_size, shuffle = False, num_workers = 8, pin_memory = True)

    # Create model
    model = HandSilhouetteNet3(mano_model_path = './models/MANO_RIGHT.pkl', num_pca_comps = args.num_pcs, device = device)
    model.to(device)

    # Evaluate model
    checkpoint = torch.load(os.path.join(args.checkpoint_path, args.checkpoint_file), map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])
    check_epoch = checkpoint['epoch']

    print('=' * 10, f'Model: {args.checkpoint_path} (Epoch: {check_epoch})', '=' * 10)

    evaluate_model(model, dataloader_frei_test, test_vertices = True, device = device)

    print()


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, default = './dataset/freihand')
    parser.add_argument('--checkpoint_path', type = str, default = './checkpoint')
    parser.add_argument('--checkpoint_file', type = str, default = 'model_pretrained.pth')
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--num_pcs', type = int, default = 45, help = 'number of pose PCs (ex: 6, 45)')
    args = parser.parse_args()

    test(args)
