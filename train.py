import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader


from dataset import FreiHandDataset
from model import HandSilhouetteNet3
from loss import criterion


def train_model(model, dataloader_train, dataloader_val, criterion, optimizer, device, num_epochs, start_epoch = 0, scheduler = None, train_loss_list = [], val_loss_list = [], last_lr_list = [], checkpoint_path = './checkpoint'):
    if start_epoch == 0:
        min_val_loss = sys.maxsize
    else:
        min_val_loss = min(val_loss_list)
    
    for epoch in range(start_epoch, num_epochs):
        train_loss, val_loss = 0, 0

        # Train Phase
        model.train()
        for inputs, focal_lens, image_refs, labels, dist_maps, meshes in dataloader_train:
            inputs = inputs.to(device)
            focal_lens = focal_lens.to(device)
            image_refs = image_refs.to(device)
            labels = labels.to(device)
            dist_maps = dist_maps.to(device)
            meshes = meshes.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, focal_lens, image_refs)
            loss = criterion(outputs, image_refs, labels, dist_maps, meshes, device)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(dataloader_train.dataset)
        train_loss_list.append(train_loss)

        # Validation Phase
        model.eval()
        with torch.no_grad():
            for inputs, focal_lens, image_refs, labels, dist_maps, meshes in dataloader_val:
                inputs = inputs.to(device)
                focal_lens = focal_lens.to(device)
                image_refs = image_refs.to(device)
                labels = labels.to(device)
                dist_maps = dist_maps.to(device)
                meshes = meshes.to(device)

                outputs = model(inputs, focal_lens, image_refs)
                loss = criterion(outputs, image_refs, labels, dist_maps, meshes, device)

                val_loss += loss.item() * inputs.size(0)
            val_loss /= len(dataloader_val.dataset)
            val_loss_list.append(val_loss)
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)
            last_lr_list.append(scheduler._last_lr)
        
        # Save the loss values
        df_loss = pd.DataFrame({'train_loss': train_loss_list, 'val_loss': val_loss_list, 'last_lr': last_lr_list})
        df_loss.to_csv(os.path.join(checkpoint_path, 'loss.csv'), index = False)

        # Save checkpoints
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, os.path.join(checkpoint_path, 'model.pth'))

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, os.path.join(checkpoint_path, f'model_epoch{epoch}.pth'))

        if (val_loss < min_val_loss):
            min_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, os.path.join(checkpoint_path, 'model_best.pth'))
        
        print(f'[Epoch {epoch}] Training Loss: {train_loss}, Validation Loss: {val_loss}, Last Learning Rate: {scheduler._last_lr}')


def main(args):
    # Configurations
    joints_anno_file = 'training_xyz.json'
    camera_Ks_file = 'training_K.json'
    data_split_file = 'FreiHand_split_ids.json'
    vertices_anno_file = 'training_verts.json'

    print('Checkpoint Path: {}\n'.format(args.checkpoint_path))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    start_epoch = 0

    # Data Loaders
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])  # based on ImageNet statistics
        transforms.Normalize(mean = [0.8705], std = [0.3358])
    ])

    dataset_train = FreiHandDataset(args.data_path, joints_anno_file, camera_Ks_file, data_split_file, vertices_anno_file, split = 'train', transform = transform, augment = True)
    dataset_val = FreiHandDataset(args.data_path, joints_anno_file, camera_Ks_file, data_split_file, vertices_anno_file, split = 'val', transform = transform, augment = False)

    dataloader_train = DataLoader(dataset = dataset_train, batch_size = args.batch_size, shuffle = True, num_workers = 16, pin_memory = True)
    dataloader_val = DataLoader(dataset = dataset_val, batch_size = args.batch_size, shuffle = False, num_workers = 8, pin_memory = True)

    print('Number of samples in training dataset: ', len(dataset_train))
    print('Number of samples in validation dataset: ', len(dataset_val))

    # Create model, optimizer, and learning rate scheduler
    model = HandSilhouetteNet3(mano_model_path = './models/MANO_RIGHT.pkl', num_pca_comps = args.num_pcs, device = device)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr = args.init_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 5, min_lr = 1e-6)

    # Train model
    train_loss_list = []
    val_loss_list = []
    last_lr_list = []

    if args.resume:
        checkpoint_file = 'model.pth'
        checkpoint = torch.load(os.path.join(args.checkpoint_path, checkpoint_file))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print('Start Epoch: {}\n'.format(start_epoch))

        df_loss = pd.read_csv(os.path.join(args.checkpoint_path, 'loss.csv'))
        train_loss_list = df_loss['train_loss'].tolist()
        val_loss_list = df_loss['val_loss'].tolist()
        last_lr_list = df_loss['last_lr'].tolist()

    train_model(model, dataloader_train, dataloader_val, criterion, optimizer, device, args.num_epochs, start_epoch, scheduler, train_loss_list, val_loss_list, last_lr_list, args.checkpoint_path)


if __name__ == '__main__':
    # Set the seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, default = './dataset/freihand')
    parser.add_argument('--checkpoint_path', type = str, default = './checkpoint')
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--num_epochs', type = int, default = 150)
    parser.add_argument('--init_lr', type = float, default = 1e-4)
    parser.add_argument('--num_pcs', type = int, default = 45, help = 'number of pose PCs (ex: 6, 45)')
    parser.add_argument('--resume', action = 'store_true')
    args = parser.parse_args()

    main(args)
