import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trimesh
import cv2
from PIL import Image
import random
import json

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from model import HandSilhouetteNet3

import mano

# PyTorch3D data structures
from pytorch3d.structures import Meshes

# PyTorch3D rendering components
from pytorch3d.renderer import (
    PerspectiveCameras, look_at_view_transform, 
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    BlendParams, SoftSilhouetteShader, TexturesVertex
)

from draw3d import save_a_image_with_mesh_joints


class FreiHandDataset(Dataset):
    def __init__(self, path, camera_Ks_file, transform = None):
        self.image_paths = []
        self.camera_Ks = []
        
        self.transform = transform

        split_ids = [11, 121, 404, 474, 1271, 1335, 1608, 1678, 3095, 3960]

        self.rgb_image_paths = [os.path.join(path, f'rgb/{i:08}.jpg') for i in split_ids]
        self.image_paths = [os.path.join(path, f'mask/{i:08}.jpg') for i in split_ids]

        with open(os.path.join(path, camera_Ks_file), 'r') as fh:
            self.camera_Ks = json.load(fh)
            self.camera_Ks = np.array(self.camera_Ks)[split_ids].tolist()
    
    def __getitem__(self, index):
        rgb_image_name = self.rgb_image_paths[index]
        rgb_image = cv2.imread(rgb_image_name)  

        image_name = self.image_paths[index]
        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)   # grayscale image 224 x 224
        
        # Binarize the image (value in {0, 255})
        image_ref = cv2.threshold(image, thresh = 127, maxval = 1, type = cv2.THRESH_BINARY)[1].astype('uint8')
        image_ref = torch.tensor(image_ref, dtype = torch.int)

        image = cv2.threshold(image, thresh = 127, maxval = 255, type = cv2.THRESH_BINARY)[1].astype('uint8')
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        focal_len = torch.tensor([self.camera_Ks[index][0][0], self.camera_Ks[index][1][1]])
        
        return rgb_image, image, focal_len, image_ref
    
    def __len__(self):
        return len(self.image_paths)


def world2pixel(sample, fx, fy, ux = 112, uy = 112):
    """
    Transform from 3D world coordinate to image coordinate
    :param sample: joints in (x, y, z) with x, y, and z in mm
    :return: joints in (u, v, d) with u, v in image coordinates and d in mm
    """
    uvd = np.zeros((len(sample), 3), np.float32)
    uvd[:, 0] = sample[:, 0] / sample[:, 2] * fx + ux
    uvd[:, 1] = sample[:, 1] / sample[:, 2] * fy + uy
    uvd[:, 2] = sample[:, 2]
    return uvd


def demo(model, dataloader, device = 'cpu'):
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
        for i, (rgb_input, inputs, focal_lens, image_refs) in enumerate(dataloader):
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


            mask = inputs.reshape(224, 224, 1).cpu().numpy()
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_0_channel = mask.reshape(224, 224)
            contour = cv2.Laplacian(mask_0_channel, -1)
            contour = cv2.threshold(contour, 0, 1, cv2.THRESH_BINARY_INV)[1]

            verts = vertices / 1000
            faces = [torch.tensor(rh_model.faces.astype(int)).to(device) for i in range(batch_size)]

            vertex2xyz = outputs['refined_joints'].cpu()
            
            camera_Ks = np.array([
                [focal_lens[0, 0].cpu(), 0., 112.],
                [0., focal_lens[0, 1].cpu(), 112.],
                [0., 0., 1.0]
            ])

            pose_uv = world2pixel(vertex2xyz[0], fx = focal_lens[0, 0].cpu(), fy = focal_lens[0, 1].cpu())[:, :2]
            pose_uv = np.array(pose_uv)
            
            vertex2xyz = vertex2xyz.reshape(21, 3).numpy().astype(float) / 1000
            save_a_image_with_mesh_joints(rgb_input[0], mask_bgr, (silhouettes[0]  == 0).cpu().numpy().astype('uint8') * 255, contour, camera_Ks, verts[0].cpu().numpy(), faces[0].cpu().numpy(), pose_uv, vertex2xyz, f'./demo_output/demo_{i}.jpg')



def test():
    os.makedirs('./demo_output', exist_ok = True)

    # Configurations
    PATH = './'

    DATA_PATH = './demo_input'
    camera_Ks_file = 'camera_K.json'

    directories = ['checkpoint/']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 1

    # Data Loaders
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.8705], std = [0.3358])
    ])

    dataset_frei_test = FreiHandDataset(DATA_PATH, camera_Ks_file, transform = transform)
    dataloader_frei_test = DataLoader(dataset = dataset_frei_test, batch_size = BATCH_SIZE, shuffle = False)

    # Create model
    model = HandSilhouetteNet3(mano_model_path = os.path.join(PATH, 'models/MANO_RIGHT.pkl'), num_pca_comps = 45, device = device)
    model.to(device)

    # Demo
    for checkpoint_dir in directories:
        checkpoint = torch.load(os.path.join(PATH, checkpoint_dir, 'model_pretrained.pth'), map_location = device)
        model.load_state_dict(checkpoint['model_state_dict'])
        check_epoch = checkpoint['epoch']

        print('=' * 10, f'Model: {checkpoint_dir} (Epoch: {check_epoch})', '=' * 10)

        demo(model, dataloader_frei_test, device = device)

        print()


if __name__ == '__main__':
    test()
