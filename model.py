import torch
import torch.nn as nn
from torchvision import models

# PyTorch3D data structures
from pytorch3d.structures import Meshes

# PyTorch3D rendering components
from pytorch3d.renderer import (
    PerspectiveCameras, RasterizationSettings, MeshRenderer,
    MeshRasterizer, BlendParams, SoftSilhouetteShader, TexturesVertex
)

import mano
from mano.lbs import vertices2joints


##################################################
## Multi-head Encoder
##################################################

class Encoder_with_Shape(nn.Module):
    def __init__(self, num_pca_comps):
        super(Encoder_with_Shape, self).__init__()

        self.feature_extractor = models.resnet18(pretrained = True)
        self.feature_extractor.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        fc_in_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Sequential(
            nn.Linear(fc_in_features, fc_in_features),
            nn.ReLU()
        )
        
        self.hand_pca_estimator = nn.Sequential(
            nn.Linear(fc_in_features, fc_in_features // 2),
            nn.ReLU(),
            nn.Linear(fc_in_features // 2, num_pca_comps)   # hand pose PCAs
        )

        self.rotation_estimator = nn.Sequential(
            nn.Linear(fc_in_features, fc_in_features // 2),
            nn.ReLU(),
            nn.Linear(fc_in_features // 2, fc_in_features // 4),
            nn.ReLU(),
            nn.Linear(fc_in_features // 4, 3)   # 3D global orientation
        )

        self.translation_estimator = nn.Sequential(
            nn.Linear(fc_in_features + 2, fc_in_features // 2),
            nn.ReLU(),
            nn.Linear(fc_in_features // 2, fc_in_features // 4),
            nn.ReLU(),
            nn.Linear(fc_in_features // 4, 3)   # 3D translation
        )

        self.hand_shape_estimator = nn.Sequential(
            nn.Linear(fc_in_features, fc_in_features // 2),
            nn.ReLU(),
            nn.Linear(fc_in_features // 2, 10)   # MANO shape parameters
        )

    def forward(self, x, focal_lens):
        x = self.feature_extractor(x)
        hand_pca = self.hand_pca_estimator(x)
        global_orientation = self.rotation_estimator(x)
        translation = self.translation_estimator(torch.cat([x, focal_lens], -1))
        hand_shape = self.hand_shape_estimator(x)
        output = torch.cat([hand_pca, global_orientation, translation, hand_shape], -1)
        return output


##################################################
## RefineNet
##################################################


class RefineNet(nn.Module):
    def __init__(self, num_vertices):
        super(RefineNet, self).__init__()

        self.num_vertices = num_vertices

        self.net = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), bias = False),
            nn.BatchNorm2d(64, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1, dilation = 1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1), bias = False),
            nn.BatchNorm2d(128, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 256, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1), bias = False),
            nn.BatchNorm2d(256, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
            nn.ReLU(inplace = True),
            nn.AdaptiveAvgPool2d(output_size = (7, 7))
        )

        self.fc = nn.Linear(7 * 7 * 256, num_vertices * 3)

    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


##################################################
## End-to-End Network
##################################################

class HandSilhouetteNet3(nn.Module):
    def __init__(self, mano_model_path, num_pca_comps, device):
        super(HandSilhouetteNet3, self).__init__()

        self.mano_model_path = mano_model_path
        self.num_pca_comps = num_pca_comps
        self.device = device

        # Encoder (from shadow to global orient & pose PCAs)
        self.encoder = Encoder_with_Shape(num_pca_comps = num_pca_comps)

        # RefineNet
        self.refine_net = RefineNet(num_vertices = 778)

        # MANO right hand template model
        self.rh_model = mano.load(
            model_path = mano_model_path,
            is_rhand = True,
            num_pca_comps = num_pca_comps,
            flat_hand_mean = True
        )

        # Configurations for rendering silhouettes

        # To blend the 100 faces we set a few parameters which control the opacity (gamma) and the sharpness (sigma) of edges
        # The sigma value determines the sharpness of the peak in the normal distribution used in the blending function.
        # When we set sigma to be smaller, the silhouette will become shaper.
        self.blend_params = BlendParams(sigma = 1e-4, gamma = 1e-4, background_color = (1.0, 1.0, 1.0))

        # Define the settings for rasterization and shading. Here we set the output image to be of size 256x256
        # To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that 
        # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
        # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
        # the difference between naive and coarse-to-fine rasterization. 
        self.raster_settings = RasterizationSettings(
            image_size = 224, 
            blur_radius = 0.0, 
            faces_per_pixel = 100, 
        )


    def forward(self, img, focal_lens, mask_gt):
        # Initialize a perspective camera
        # fx = fx_screen * 2.0 / image_width
        # fy = fy_screen * 2.0 / image_height
        # px = - (px_screen - image_width / 2.0) * 2.0 / image_width
        # py = - (py_screen - image_height / 2.0) * 2.0 / image_height
        self.cameras = PerspectiveCameras(
            focal_length = focal_lens * 2.0 / 224,
            device = self.device
        )

        # Create a silhouette mesh renderer by composing a rasterizer and a shader
        self.silhouette_renderer = MeshRenderer(
            rasterizer = MeshRasterizer(
                cameras = self.cameras, 
                raster_settings = self.raster_settings
            ),
            shader = SoftSilhouetteShader(blend_params = self.blend_params)
        )

        # Silhouette to pose PCAs & 3D global orientation & 3D translation & shape parameters
        code = self.encoder(img, focal_lens)
        pose_pcas, global_orient, transl, betas = code[:, :-16], code[:, -16:-13], code[:, -13:-10], code[:, -10:]

        batch_size = code.shape[0]

        # Global orient & pose PCAs to 3D hand joints & reconstructed silhouette
        rh_output = self.rh_model(
            betas = betas,
            global_orient = global_orient,
            hand_pose = pose_pcas,
            transl = transl,
            return_verts = True,
            return_tips = True
        )
        
        # Initialize each vertex to be white in color
        verts_rgb = torch.ones_like(rh_output.vertices)  # (B, V, 3)
        textures = TexturesVertex(verts_features = verts_rgb.to(self.device))

        # Coordinate transformation from FreiHand to PyTorch3D for rendering
        # [FreiHand] +X: right, +Y: down, +Z: in
        # [PyTorch3D] +X: left, +Y: up, +Z: in
        coordinate_transform = torch.tensor([[-1, -1, 1]]).to(self.device)

        mesh_faces = torch.tensor(self.rh_model.faces.astype(int)).to(self.device)

        # Create a Meshes object
        hand_meshes = Meshes(
            verts = [rh_output.vertices[i] * coordinate_transform for i in range(batch_size)],
            faces = [mesh_faces for i in range(batch_size)],
            textures = textures
        )

        # Render the meshes
        silhouettes = self.silhouette_renderer(meshes_world = hand_meshes)
        silhouettes = silhouettes[..., 3]

        # Reorder the joints to match FreiHand annotations
        reorder = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
        output_joints = rh_output.joints[:, reorder, :]

        #################### Refinement Start ####################
        diff_map = torch.cat((mask_gt.unsqueeze(1), silhouettes.unsqueeze(1)), dim = 1)
        offset = self.refine_net(diff_map)
        offset = torch.clamp(offset, min = -50, max = 50)
        offset = offset.view(-1, 778, 3)

        vertices = rh_output.vertices + offset

        refined_joints = vertices2joints(self.rh_model.J_regressor, vertices)
        refined_joints = self.rh_model.add_joints(vertices, refined_joints)[:, reorder, :]
        #################### Refinement End ######################

        result = {
            'code': code,
            'joints': output_joints,
            'silhouettes': silhouettes,
            'vertices': rh_output.vertices,
            'refined_joints': refined_joints,
            'refined_vertices': vertices,
            'betas': betas
        }

        return result
