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
from eval_utils import DiceCoeff

import mano

# PyTorch3D data structures
from pytorch3d.structures import Meshes

# PyTorch3D rendering components
from pytorch3d.renderer import (
    PerspectiveCameras, look_at_view_transform, 
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    BlendParams, SoftSilhouetteShader, TexturesVertex
)


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def evaluate_model(model, dataloader, test_vertices = True, device = 'cpu'):
    # Initialize evaluation utilities
    miou_evaluator = Evaluator(num_class = 2)

    dice = DiceCoeff()
    dice_mean = 0

    rh_model = mano.load(
        model_path = './models/MANO_RIGHT.pkl',
        is_rhand = True,
        num_pca_comps = 45,
        flat_hand_mean = True
    )

    # Renderer settings
    blend_params = BlendParams(sigma = 1e-4, gamma = 1e-4, background_color = (1.0, 1.0, 1.0))

    raster_settings = RasterizationSettings(
        image_size = 224, 
        blur_radius = 0.0, 
        faces_per_pixel = 100, 
    )

    # Evaluation phase
    model.eval()
    with torch.no_grad():
        # Iterate over the dataset once
        for i, (inputs, focal_lens, image_refs, joints_anno, dist_maps, meshes_anno) in enumerate(dataloader):
            inputs = inputs.to(device)
            focal_lens = focal_lens.to(device)
            image_refs = image_refs.to(device)

            outputs = model(inputs, focal_lens, image_refs)

            vertices = outputs['refined_vertices']

            batch_size = vertices.shape[0]

            # Re-render silhouettes
            cameras = PerspectiveCameras(
                focal_length = focal_lens * 2.0 / 224,
                device = device
            )

            silhouette_renderer = MeshRenderer(
                rasterizer = MeshRasterizer(
                    cameras = cameras, 
                    raster_settings = raster_settings
                ),
                shader = SoftSilhouetteShader(blend_params = blend_params)
            )

            verts_rgb = torch.ones_like(vertices)  # (B, V, 3)
            textures = TexturesVertex(verts_features = verts_rgb.to(device))

            # Coordinate transformation from FreiHand to PyTorch3D for rendering
            # [FreiHand] +X: right, +Y: down, +Z: in
            # [PyTorch3D] +X: left, +Y: up, +Z: in
            coordinate_transform = torch.tensor([[-1, -1, 1]]).to(device)

            # Create a Meshes object
            hand_meshes = Meshes(
                verts = [(vertices[i] * coordinate_transform).float().to(device) for i in range(batch_size)],
                faces = [torch.tensor(rh_model.faces.astype(int)).to(device) for i in range(batch_size)],
                textures = textures
            )

            silhouettes = silhouette_renderer(meshes_world = hand_meshes)
            silhouettes = silhouettes[..., 3]

            # Dice
            dice_mean += dice(silhouettes, image_refs).item() * inputs.size(0)

            miou_evaluator.add_batch(image_refs[0].cpu().numpy(), (silhouettes[0].cpu().numpy() != 0))
            

        # Calculate results
        dice_mean /= len(dataloader.dataset)

        miou = miou_evaluator.Mean_Intersection_over_Union()

        print('[Evaluation of Re-projected Silhouettes]')
        print(f'mIoU = {miou}')
        print(f'Dice = {dice_mean}')


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
