import numpy as np
import cv2
import torch
import torch.nn as nn
from kornia.geometry.camera import project_points
from kornia.geometry.conversions import angle_axis_to_rotation_matrix


class CameraTransform(nn.Module):
    """
    pinhole perspective camera model with intrinsic parameters C
    and external parameters rvec and tvec
    """
    def __init__(self, num_points, device):
        super(CameraTransform, self).__init__()
        self.C = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32, device=device)
        self.num_points = num_points

    def forward(self, points, rvec, tvec):
        batch_size = rvec.shape[0]
        # first transform points from canonical coord system into camera frame
        R_c = angle_axis_to_rotation_matrix(rvec)
        R_c = R_c.unsqueeze(1).repeat(1, self.num_points, 1, 1)
        t_c = tvec.unsqueeze(1).repeat(1, self.num_points, 1, 1).view(batch_size, self.num_points, 3, 1)
        cam_points = R_c @ points.unsqueeze(-1) + t_c
        # project via fixed intrinsic camera parameters
        projected = project_points(cam_points.view(-1, 3), self.C.view(1, 3, 3))

        return cam_points, projected.view(batch_size, self.num_points, 2)


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


class SpatialTransformer(nn.Module):
    """
    for details see https://arxiv.org/abs/2006.14107
    """
    def __init__(self, device):
        super(SpatialTransformer, self).__init__()
        self.device = device
        self.v2 = torch.tensor([0, 1], device=self.device, dtype=torch.float32).view(1, 1, 2)
        self.indices1 = torch.tensor([1, 2, 0, 1, 0, 3, 4, 5, 2, 3, 8, 9, 6, 7, 10, 11])
        self.indices2 = torch.tensor([0, 3, 4, 5, 2, 1, 8, 9, 6, 7, 12, 13, 10, 11, 14, 15])

    def get_angles(self, v):
        epsilon = 1e-7
        shape = v.shape
        v1 = v
        v2 = self.v2.repeat(shape[0], shape[1], 1)
        vec_prod = v1 @ v2.permute(0, 2, 1)
        inv = vec_prod / (torch.norm(v1, dim=-1, keepdim=True) * torch.norm(v2, dim=-1, keepdim=True))
        angle = torch.acos(torch.clamp(inv, -1 + epsilon, 1 - epsilon))

        return angle

    def get_gaussian_parameters(self, points):

        v = points[:, self.indices2] - points[:, self.indices1]

        centers = points[:, self.indices1] + v / 2

        # hacky way to make sure angles/vectors are in right orientation
        vx = v[..., 0]
        vy = v[..., 1]

        vy[vx < 0] = vy[vx < 0] * -1
        vx[vx < 0] = vx[vx < 0] * -1

        v[..., 0] = vx
        v[..., 1] = vy

        angles = self.get_angles(v)

        lengths = torch.norm(v, dim=-1)

        return centers, angles, lengths

    def draw_gaussian(self, center, angle, sigma_x, sigma_y, size, device):
        x, y = torch.meshgrid(torch.arange(0, size), torch.arange(0, size))
        x, y = x.unsqueeze(0).unsqueeze(0).to(device), \
               y.unsqueeze(0).unsqueeze(0).to(device)

        angle = angle.to(device)
        a = torch.cos(angle) ** 2 / (2 * sigma_x ** 2) + torch.sin(angle) ** 2 / (2 * sigma_y ** 2)
        b = -torch.sin(2 * angle) / (4 * sigma_x ** 2) + torch.sin(2 * angle) / (4 * sigma_y ** 2)
        c = torch.sin(angle) ** 2 / (2 * sigma_x ** 2) + torch.cos(angle) ** 2 / (2 * sigma_y ** 2)

        a = a.unsqueeze(-1).unsqueeze(-1)

        b = b.unsqueeze(-1).unsqueeze(-1)

        c = c.unsqueeze(-1).unsqueeze(-1)

        # we clamp values here to avoid numerical instability due to very low numbers close to float32 range

        xdist = torch.clamp((x - center[:, :, 0]), -500, 500)

        ydist = torch.clamp((y - center[:, :, 1]), -500, 500)

        g = torch.exp((-a * xdist ** 2 - 2 * b * xdist * ydist - c * ydist ** 2))

        return g

    def forward(self, points):
        centers, angles, lengths = self.get_gaussian_parameters(points)
        centers = centers.view(points.shape[0], centers.shape[1], 2, 1, 1)
        angles = angles[..., 0]
        s_y = lengths / 3.7
        s_x = torch.ones_like(s_y)
        return self.draw_gaussian(centers, angles, s_x, s_y, size=64, device=self.device)