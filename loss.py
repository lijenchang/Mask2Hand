import torch
import torch.nn as nn
import torch.nn.functional as F

from mano.lbs import batch_rodrigues


class IoULoss(nn.Module):
    def __init__(self, smooth = 1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, labels):
        batch_size = outputs.shape[0]

        outputs = outputs.view(batch_size, -1)
        labels = labels.view(batch_size, -1)

        intersection = (outputs * labels).sum(1)
        union = (outputs + labels).sum(1) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)  # Smooth the division to avoid 0/0
        iou = iou.mean()
        
        return 1 - iou   # or -torch.log(iou)


class DiceLoss(nn.Module):
    def __init__(self, smooth = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, labels):
        batch_size = outputs.shape[0]

        outputs = outputs.view(batch_size, -1)
        labels = labels.view(batch_size, -1)

        intersection = (outputs * labels).sum(1)
        summation = (outputs + labels).sum(1)

        dice = (2 * intersection + self.smooth) / (summation + self.smooth)  # Smooth the division to avoid 0/0
        dice = dice.mean()
        
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                'none': No reduction will be applied to the output.
                'mean': The output will be averaged.
                'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    def __init__(self, alpha = -1., gamma = 2., reduction = 'none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        #p = torch.sigmoid(inputs)
        p = inputs
        ce_loss = F.binary_cross_entropy(inputs, targets.float(), reduction = 'none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class ContourLoss(nn.Module):
    def __init__(self, device):
        super(ContourLoss, self).__init__()
        self.device = device
    
    def forward(self, outputs, dist_maps):
        # Binarize outputs [0.0, 1.0] -> {0., 1.}
        # outputs = (outputs >= 0.5).float()   # Thresholding is NOT differentiable
        outputs = 1 / (1 + torch.exp(-100 * (outputs - 0.5)))   # Differentiable binarization (approximation)
        mask = (outputs < 0.5)
        outputs = outputs * mask   # Zero out values above the threshold 0.5

        # Convert from (B x H x W) to (B x C x H x W)
        outputs = torch.unsqueeze(outputs, 1)

        # Apply Laplacian operator to grayscale images to find contours
        kernel = torch.tensor([[[
            [ 0.,  1.,  0.],
            [ 1., -4.,  1.],
            [ 0.,  1.,  0.]
        ]]]).to(self.device)

        contours = F.conv2d(outputs, kernel, padding = 1)
        contours = torch.clamp(contours, min = 0, max = 255)

        # Convert from (B x C x H x W) back to (B x H x W)
        contours = torch.squeeze(contours, 1)

        # Compute the Chamfer distance between two images
        # Selecting indices is NOT differentiable -> use tanh(x) or 2 / (1 + e^(-100(x))) - 1 for differentiable thresholding
        # -> apply element-wise product between contours and distance maps
        contours = torch.tanh(contours)
        dist = contours * dist_maps   # element-wise product

        dist = dist.sum() / contours.shape[0]
        assert(dist >= 0)

        return dist


class GeodesicDistance(nn.Module):
    r"""
    Calculate the geodesic distance between rotation matrices.
    Borrowed from https://github.com/airalcorn2/pytorch-geodesic-loss.
    See also: https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html#pytorch3d.transforms.so3_relative_angle.
    The distance ranges from 0 to :math:`pi`.
    Both `input` and `target` consist of rotation matrices, i.e., they have to be Tensors
    of size :math:`(minibatch, 3, 3)`.
    The loss can be described as:
    .. math::
        \text{loss}(R_{S}, R_{T}) = \arccos\left(\frac{\text{tr} (R_{S} R_{T}^{T}) - 1}{2}\right)
    Args:
        eps (float, optional): term to improve numerical stability (default: 1e-7). See:
            https://github.com/pytorch/pytorch/issues/8069.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``
    Shape:
        - Input: :math:`(N, 3, 3)`.
        - Target: :math:`(N, 3, 3)`.
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then :math:`(N)`.
    """    
    def __init__(self, eps: float = 1e-7, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        R_diffs = input @ target.permute(0, 2, 1)
        # See: https://github.com/pytorch/pytorch/issues/7500#issuecomment-502122839.
        traces = R_diffs.diagonal(dim1 = -2, dim2 = -1).sum(-1)
        dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + self.eps, 1 - self.eps))
        if self.reduction == "none":
            return dists
        elif self.reduction == "mean":
            return dists.mean()
        elif self.reduction == "sum":
            return dists.sum()


def camera_pose_loss(cam1_R, cam1_T, cam2_R, cam2_T):
    """
    Calculates the divergence of a batch of pairs of cameras cam1, cam2.
    This is measured by the weighted sum of the geodesic distance between
    rotation components of the camera extrinsics and the L2 distance
    between translation vectors.
    Ref: https://pytorch3d.org/tutorials/bundle_adjustment
    """
    # rotation distance
    R_distance = GeodesicDistance(reduction = 'mean')(cam1_R, cam2_R)
    # translation distance
    T_distance = ((cam1_T - cam2_T) ** 2).sum(1).mean()
    # weighted sum
    return R_distance + 1e-2 * T_distance


def batch_align_w_scale(mtx1, mtx2):
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    t1 = torch.mean(mtx1, dim = 1, keepdim = True)
    t2 = torch.mean(mtx2, dim = 1, keepdim = True)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = torch.norm(mtx1_t, dim = (1, 2), keepdim = True) + 1e-8
    mtx1_t = mtx1_t / s1
    s2 = torch.norm(mtx2_t, dim = (1, 2), keepdim = True) + 1e-8
    mtx2_t = mtx2_t / s2

    # orthogonal procrustes alignment
    # u, w, vt = torch.linalg.svd(torch.bmm(torch.transpose(mtx2_t, 1, 2), mtx1_t).transpose(1, 2))
    # R = torch.bmm(u, vt)
    u, w, v = torch.svd(torch.bmm(mtx2_t.transpose(1, 2), mtx1_t).transpose(1, 2))
    R = torch.bmm(u, v.transpose(1, 2))
    s = w.sum(dim = 1, keepdim = True)

    # apply trafos to the second matrix
    mtx2_t = torch.bmm(mtx2_t, R.transpose(1, 2)) * s.unsqueeze(1)
    mtx2_t = mtx2_t * s1 + t1
    
    return mtx2_t


def aligned_joints_loss(joints_gt, joints_pred):
    joints_pred_aligned = batch_align_w_scale(joints_gt, joints_pred)
    return nn.MSELoss()(joints_pred_aligned, joints_gt)


def aligned_meshes_loss(meshes_gt, meshes_pred):
    meshes_pred_aligned = batch_align_w_scale(meshes_gt, meshes_pred)
    return nn.L1Loss()(meshes_pred_aligned, meshes_gt)


def criterion(outputs, image_refs, labels, dist_maps, meshes, device):
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    contour_loss = ContourLoss(device = device)

    '''
    return 2e-3 * mse_loss(outputs['joints'], labels) \
        + 2e-2 * aligned_joints_loss(labels, outputs['joints']) \
        + 2e-3 * mse_loss(outputs['refined_joints'], labels) \
        + 2e-2 * aligned_joints_loss(labels, outputs['refined_joints']) \
        + 0.1 * l1_loss(outputs['refined_vertices'], meshes) \
        + aligned_meshes_loss(meshes, outputs['refined_vertices']) \
        + 0.5 * F.binary_cross_entropy(outputs['silhouettes'], (image_refs).float()) \
        + 1e-4 * contour_loss(outputs['silhouettes'], dist_maps)
    '''
    return 2e-3 * mse_loss(outputs['joints'], labels) \
        + 2e-2 * aligned_joints_loss(labels, outputs['joints']) \
        + 2e-3 * mse_loss(outputs['refined_joints'], labels) \
        + 2e-2 * aligned_joints_loss(labels, outputs['refined_joints']) \
        + 0.1 * l1_loss(outputs['refined_vertices'], meshes) \
        + aligned_meshes_loss(meshes, outputs['refined_vertices']) \
        + 1e-4 * contour_loss(outputs['silhouettes'], dist_maps)

