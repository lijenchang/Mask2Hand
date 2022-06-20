import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import random
import json
import torch
from torch.utils.data import Dataset


##################################################
## Synthetic MANO Dataset
##################################################

class HandPoseDataset(Dataset):
    def __init__(self, path, label_file, transform = None, augment = False):
        self.image_paths = []
        self.global_orients = []
        self.translations = []
        self.pose_pcas = []
        self.joints = []
        self.focal_lengths = []
        self.vertices = []
        
        self.transform = transform
        self.augment = augment
        
        df = pd.read_csv(label_file)
        import json
        for i in df.index:
            self.image_paths.append(os.path.join(path, str(df.loc[i, 'image_path'])))
            self.global_orients.append(json.loads(df.loc[i, 'global_orient']))
            self.translations.append(json.loads(df.loc[i, 'translation']))
            self.pose_pcas.append(json.loads(df.loc[i, 'pose_pca']))
            self.joints.append(json.loads(df.loc[i, 'joint']))
            self.focal_lengths.append(json.loads(df.loc[i, 'focal_length']))
            self.vertices.append(json.loads(df.loc[i, 'vertices']))
            #if i == 19999:
            #    break
    
    def __getitem__(self, index):
        if self.augment:
            rand = random.choice([0, 1, 2, 3, 4])
            index = 5 * index + rand

        image_name = self.image_paths[index]
        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)   # binary image with value {0, 255}
        image = 255 - image

        if image.shape != (224, 224):
            image_ref = cv2.resize(image, (224, 224))
            image_ref = cv2.threshold(image_ref, 127, 1, cv2.THRESH_BINARY)[1]
        else:
            image_ref = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)[1]

        # Extract contour and compute distance transform
        contour = cv2.Laplacian(image_ref, -1)
        contour = cv2.threshold(contour, 0, 1, cv2.THRESH_BINARY_INV)[1]
        dist_map = cv2.distanceTransform(contour, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        dist_map = torch.tensor(dist_map)

        image_ref = torch.tensor(image_ref, dtype = torch.int)
        
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.joints[index])
        mesh = torch.tensor(self.vertices[index])
        
        focal_len = torch.tensor(self.focal_lengths[index])

        return image, focal_len, image_ref, label, dist_map, mesh
    
    def __len__(self):
        if self.augment:
            return len(self.image_paths) // 5
        else:
            return len(self.image_paths)


##################################################
## FreiHAND Dataset
##################################################

class FreiHandDataset(Dataset):
    def __init__(self, path, joints_anno_file, camera_Ks_file, data_split_file, vertices_anno_file, split = 'train', transform = None, augment = False):
        self.image_paths = []
        self.joints = []
        self.camera_Ks = []
        self.vertices = []
        
        self.transform = transform
        self.augment = augment

        with open(os.path.join(path, data_split_file), 'r') as fh:
            split_ids = json.load(fh)[f'{split}_ids']
        
        with open(os.path.join(path, joints_anno_file), 'r') as fh:
            self.joints = json.load(fh)
            self.joints = (np.array(self.joints)[split_ids] * 1000).tolist()

        self.image_paths = [os.path.join(path, f'training/mask/{i:08}.jpg') for i in split_ids]

        with open(os.path.join(path, camera_Ks_file), 'r') as fh:
            self.camera_Ks = json.load(fh)
            self.camera_Ks = np.array(self.camera_Ks)[split_ids].tolist()
        
        with open(os.path.join(path, vertices_anno_file), 'r') as fh:
            self.vertices = json.load(fh)
            self.vertices = (np.array(self.vertices)[split_ids] * 1000).tolist()
    
    def __getitem__(self, index):
        image_name = self.image_paths[index]
        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)   # grayscale image 224 x 224

        if self.augment:
            # label in (u, v, d) coordinate
            label_uvd = self.world2pixel(np.array(self.joints[index]), fx = self.camera_Ks[index][0][0], fy = self.camera_Ks[index][1][1])

            mesh_uvd = self.world2pixel(np.array(self.vertices[index]), fx = self.camera_Ks[index][0][0], fy = self.camera_Ks[index][1][1])
            
            # rotation and scaling
            RandomRotate = np.random.randint(-180, 180)
            RandomScale = 0.2 * np.random.rand() + 0.9
            matrix = cv2.getRotationMatrix2D((224 / 2, 224 / 2), RandomRotate, RandomScale)
            
            image, label_uvd[:, :2], mesh_uvd[:, :2] = self.augment_transform(image, label_uvd[:, :2], mesh_uvd[:, :2], matrix)
            
            # label in (x, y, z) coordinate
            label_xyz = self.pixel2world(label_uvd, fx = self.camera_Ks[index][0][0], fy = self.camera_Ks[index][1][1])

            mesh_xyz = self.pixel2world(mesh_uvd, fx = self.camera_Ks[index][0][0], fy = self.camera_Ks[index][1][1])
        else:
            label_xyz = self.joints[index]
            mesh_xyz = self.vertices[index]
        
        label = torch.tensor(label_xyz)
        mesh = torch.tensor(mesh_xyz)

        # Binarize the image (value in {0, 255})
        image_ref = cv2.threshold(image, thresh = 127, maxval = 1, type = cv2.THRESH_BINARY)[1].astype('uint8')

        image = cv2.threshold(image, thresh = 127, maxval = 255, type = cv2.THRESH_BINARY)[1].astype('uint8')

        # Extract contour and compute distance transform
        contour = cv2.Laplacian(image_ref, -1)
        contour = cv2.threshold(contour, 0, 1, cv2.THRESH_BINARY_INV)[1]
        dist_map = cv2.distanceTransform(contour, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        dist_map = torch.tensor(dist_map)

        image_ref = torch.tensor(image_ref, dtype = torch.int)
        
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        focal_len = torch.tensor([self.camera_Ks[index][0][0], self.camera_Ks[index][1][1]])
        
        return image, focal_len, image_ref, label, dist_map, mesh
    
    def augment_transform(self, img, label, mesh, matrix):
        '''
        img: [H, W]  label, [N, 2]   
        '''
        img_out = cv2.warpAffine(img, matrix, (224, 224))
        label_out = np.ones((21, 3))
        label_out[:, :2] = label[:, :2].copy()
        label_out = np.matmul(matrix, label_out.transpose())
        label_out = label_out.transpose()

        mesh_out = np.ones((778, 3))
        mesh_out[:, :2] = mesh[:, :2].copy()
        mesh_out = np.matmul(matrix, mesh_out.transpose())
        mesh_out = mesh_out.transpose()

        return img_out, label_out, mesh_out
    
    def world2pixel(self, sample, fx, fy, ux = 112, uy = 112):
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
    
    def pixel2world(self, sample, fx, fy, ux = 112, uy = 112):
        """
        Transform from image coordinate to 3D world coordinate
        :param sample: joints in (u, v, d)
        :return: joints in (x, y, z) in mm
        """
        xyz = np.zeros((len(sample), 3), np.float32)
        xyz[:, 0] = ((sample[:, 0] - ux) * sample[:, 2]) / fx
        xyz[:, 1] = ((sample[:, 1] - uy) * sample[:, 2]) / fy
        xyz[:, 2] = sample[:, 2]
        return xyz
    
    def __len__(self):
        return len(self.image_paths)


##################################################
## FreiHAND Dataset with Estimated Masks
##################################################

class FreiHandDataset_Estimated(Dataset):
    def __init__(self, path, joints_anno_file, camera_Ks_file, data_split_file, vertices_anno_file, split = 'train', transform = None, augment = False):
        self.image_paths = []
        self.joints = []
        self.camera_Ks = []
        self.vertices = []
        
        self.transform = transform
        self.augment = augment

        with open(os.path.join(path, data_split_file), 'r') as fh:
            split_ids = json.load(fh)[f'{split}_ids']
        
        with open(os.path.join(path, joints_anno_file), 'r') as fh:
            self.joints = json.load(fh)
            self.joints = (np.array(self.joints)[split_ids] * 1000).tolist()

        self.image_paths = [os.path.join(path, f'training/mask/{i:08}.jpg') for i in split_ids]

        with open(os.path.join(path, camera_Ks_file), 'r') as fh:
            self.camera_Ks = json.load(fh)
            self.camera_Ks = np.array(self.camera_Ks)[split_ids].tolist()
        
        with open(os.path.join(path, vertices_anno_file), 'r') as fh:
            self.vertices = json.load(fh)
            self.vertices = (np.array(self.vertices)[split_ids] * 1000).tolist()
    
    def __getitem__(self, index):
        image_name = self.image_paths[index]
        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)   # grayscale image 224 x 224

        if self.augment:
            # label in (u, v, d) coordinate
            label_uvd = self.world2pixel(np.array(self.joints[index]), fx = self.camera_Ks[index][0][0], fy = self.camera_Ks[index][1][1])

            mesh_uvd = self.world2pixel(np.array(self.vertices[index]), fx = self.camera_Ks[index][0][0], fy = self.camera_Ks[index][1][1])
            
            # rotation and scaling
            RandomRotate = np.random.randint(-180, 180)
            RandomScale = 0.2 * np.random.rand() + 0.9
            matrix = cv2.getRotationMatrix2D((224 / 2, 224 / 2), RandomRotate, RandomScale)
            
            image, label_uvd[:, :2], mesh_uvd[:, :2] = self.augment_transform(image, label_uvd[:, :2], mesh_uvd[:, :2], matrix)
            
            # label in (x, y, z) coordinate
            label_xyz = self.pixel2world(label_uvd, fx = self.camera_Ks[index][0][0], fy = self.camera_Ks[index][1][1])

            mesh_xyz = self.pixel2world(mesh_uvd, fx = self.camera_Ks[index][0][0], fy = self.camera_Ks[index][1][1])
        else:
            label_xyz = self.joints[index]
            mesh_xyz = self.vertices[index]
        
        label = torch.tensor(label_xyz)
        mesh = torch.tensor(mesh_xyz)

        # Binarize the image (value in {0, 255})
        image_ref = cv2.threshold(image, thresh = 127, maxval = 1, type = cv2.THRESH_BINARY)[1].astype('uint8')

        image = cv2.threshold(image, thresh = 127, maxval = 255, type = cv2.THRESH_BINARY)[1].astype('uint8')

        # Extract contour and compute distance transform
        contour = cv2.Laplacian(image_ref, -1)
        contour = cv2.threshold(contour, 0, 1, cv2.THRESH_BINARY_INV)[1]
        dist_map = cv2.distanceTransform(contour, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        dist_map = torch.tensor(dist_map)

        image_ref = torch.tensor(image_ref, dtype = torch.int)
        
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        focal_len = torch.tensor([self.camera_Ks[index][0][0], self.camera_Ks[index][1][1]])
        
        return image, focal_len, image_ref, label, dist_map, mesh
    
    def augment_transform(self, img, label, mesh, matrix):
        '''
        img: [H, W]  label, [N, 2]   
        '''
        img_out = cv2.warpAffine(img, matrix, (224, 224))
        label_out = np.ones((21, 3))
        label_out[:, :2] = label[:, :2].copy()
        label_out = np.matmul(matrix, label_out.transpose())
        label_out = label_out.transpose()

        mesh_out = np.ones((778, 3))
        mesh_out[:, :2] = mesh[:, :2].copy()
        mesh_out = np.matmul(matrix, mesh_out.transpose())
        mesh_out = mesh_out.transpose()

        return img_out, label_out, mesh_out
    
    def world2pixel(self, sample, fx, fy, ux = 112, uy = 112):
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
    
    def pixel2world(self, sample, fx, fy, ux = 112, uy = 112):
        """
        Transform from image coordinate to 3D world coordinate
        :param sample: joints in (u, v, d)
        :return: joints in (x, y, z) in mm
        """
        xyz = np.zeros((len(sample), 3), np.float32)
        xyz[:, 0] = ((sample[:, 0] - ux) * sample[:, 2]) / fx
        xyz[:, 1] = ((sample[:, 1] - uy) * sample[:, 2]) / fy
        xyz[:, 2] = sample[:, 2]
        return xyz
    
    def __len__(self):
        return len(self.image_paths)
