import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class IoU(nn.Module):
    def __init__(self):
        super(IoU, self).__init__()

    def forward(self, outputs, image_refs): # calculate IoU for pixels with value 0
        EPS = 1e-6
        
        outputs = (outputs != 0)
        image_refs = (image_refs != 0)

        intersection = (outputs & image_refs).float().sum((1, 2))
        union = (outputs | image_refs).float().sum((1, 2))

        iou = (intersection + EPS) / (union + EPS)  # Smooth the division to avoid 0/0
        
        return iou.mean()


class DiceCoeff(nn.Module):
    def __init__(self, smooth = 1e-6):
        super(DiceCoeff, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, image_refs): # calculate dice coefficient for pixels with value 0
        outputs = (outputs != 0)
        image_refs = (image_refs != 0)
        
        intersection = (outputs & image_refs).float().sum((1, 2))
        summation = (outputs.float() + image_refs.float()).sum((1, 2))

        dice = (2 * intersection + self.smooth) / (summation + self.smooth)  # Smooth the division to avoid 0/0
        dice = dice.mean()
        
        return dice


class ChamferDistance(nn.Module):
    def __init__(self, device):
        super(ChamferDistance, self).__init__()
        self.device = device
    
    def forward(self, outputs, dist_maps):
        # Binarize outputs [0.0, 1.0] -> {0., 1.}
        outputs = (outputs >= 0.5).float()

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
        contours = (contours > 0)
        dist = contours * dist_maps   # element-wise product

        dist = dist.sum() / contours.shape[0]
        assert(dist >= 0)

        return dist


def chamfer_distance(output, image_ref):
    contour = cv2.Laplacian(image_ref, -1)
    contour = cv2.threshold(contour, 0, 1, cv2.THRESH_BINARY_INV)[1]
    dist_map = cv2.distanceTransform(contour, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    contour = cv2.Laplacian(output, -1)
    contour = cv2.threshold(contour, 0, 1, cv2.THRESH_BINARY)[1]

    dist = (contour * dist_map).sum()

    return dist


class EvalUtil:
    """
    Utility class for evaluation.
    Modified from https://github.com/lmb-freiburg/freihand/blob/master/utils/eval_util.py
    """
    def __init__(self, num_kp = 21):
        # init empty data storage
        self.data = list()
        self.num_kp = num_kp
        self.data = [list() for _ in range(num_kp)]

    def feed(self, keypoint_gt, keypoint_pred):
        """ Used to feed data to the class. Stores the euclidean distance between gt and pred. """
        assert len(keypoint_gt.shape) == 3
        assert len(keypoint_pred.shape) == 3

        # calculate euclidean distance
        diff = keypoint_gt - keypoint_pred
        euclidean_dist = torch.sqrt(torch.sum(diff ** 2, 2))

        batch_size = keypoint_gt.shape[0]
        num_kp = keypoint_gt.shape[1]
        
        for i in range(num_kp):
            for j in range(batch_size):
                self.data[i].append(euclidean_dist[j][i].cpu().data.item())

    def _get_pck(self, kp_id, threshold):
        """ Returns PCK for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype('float'))
        return pck

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None, None

        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median

    def get_measures(self, val_min, val_max, steps):
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()

        # Create one plot for each part
        for part_id in range(self.num_kp):
            # mean/median error
            mean, median = self._get_epe(part_id)

            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)

        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(np.array(pck_curve_all), 0)  # mean only over keypoints

        return epe_mean_all, epe_median_all, auc_all, pck_curve_all, thresholds


def align_w_scale(mtx1, mtx2, return_trafo = False):
    """ Align the predicted entity in some optimality sense with the ground truth. """
    from scipy.linalg import orthogonal_procrustes
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # apply trafos to the second matrix
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1
    if return_trafo:
        return R, s, s1, t1 - t2
    else:
        return mtx2_t
