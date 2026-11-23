# PD-BEV port (Gen2E): new file. PCCRDetDataset plus the inputs PD-BEV (domain generalisation) trains with.
"""PD-BEV data on the co-author's Megvii PCCR base (PCCRDetDataset: images, ida / bda augmentation, GT in the ego
frame; PD-BEV uses no LiDAR depth, so return_depth stays False). Added per sample, as the official PD-BEV pipeline
computes them (our fixed copy: bevdepth/mmdet3d/datasets/pipelines/loading_UDA.py, PrepareImageInputs_UDA and
Load3DBoxesHeatmap; fixes in docs/bug_ledger.md):

- train and test: the actual intrinsics of every camera after its ida augmentation (resize, crop; PDBEV-9: a
  horizontal flip mirrors the principal point; the rotation stays out, as official). The virtual depth and the
  31-dim camera input of the DepthNet use them;
- train only: one virtual camera per real camera, PD-BEV's extrinsic perturbation (ZYX Euler angles of sensor2ego
  plus uniform noise with the PDBEV-10 axis mapping, position plus uniform noise in the ego frame; for a key frame the
  official sensor2keyego_aug is this sensor2ego, PDBEV-3), and the 2D targets of the real and the virtual views at
  1/8 of the network input: class heatmaps, the official 3x3 (real) / 2x3 (virtual) regression windows with the
  sub-pixel centre offset, virtual centre depth, box size (dx, dy, dz) and metric centre depth. The objects are the
  ones the 3D head trains on (PDBEV-4/5: PCCRDetDataset.get_gt's boxes, centre inside the BEV range after the bda
  augmentation); validity is checked on the augmented feature-map coordinates (PDBEV-6/7).
"""
from math import asin, atan2, cos, sin

import numpy as np
import torch
from mmdet3d.core import draw_heatmap_gaussian, gaussian_radius
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from pyquaternion import Quaternion

from bevdepth.datasets.pccr_det_dataset import PCCRDetDataset
from bevdepth.datasets.pccr_det_dataset import collate_fn as pccr_collate_fn

__all__ = ['PCCRPDBEVDataset', 'collate_fn_pdbev', 'PDBEV_AUG', 'virtual_depth_scale']

# virtual-camera perturbation ranges (rad, m), as the old-base configs bevdepth/configs/gen2e/pdbev-r50-pccr-{A,B}.py
PDBEV_AUG = {
    # A = official DG config (configs/PDBEV/pdbev-r50-cbgs-NUS2X-dg.py)
    'A': dict(pitch_aug=(-0.04, 0.04), yaw_aug=(-0.4, 0.4), roll_aug=(-0.04, 0.04),
              extri_x_aug=(-2.0, 2.0), extri_y_aug=(-2.0, 2.0), extri_z_aug=(-2.0, 2.0)),
    # B = the camera changes of the PCCR test rigs (decision 2026-11-22; uses knowledge of the test rigs)
    'B': dict(pitch_aug=(-0.3491, 0.3491), yaw_aug=(-0.3927, 0.3927), roll_aug=(-0.04, 0.04),
              extri_x_aug=(-2.9, 5.3), extri_y_aug=(-0.6, 0.8), extri_z_aug=(-0.5, 2.7)),
}
F_REF = 450.0  # PD-BEV's reference focal length: c = sqrt(100 / 450^2 + 100 / 450^2)
C_REF = float(np.sqrt(100.0 / np.square(F_REF) + 100.0 / np.square(F_REF)))


def virtual_depth_scale(fx, fy):
    """virtual depth = metric depth * scale, with the actual (augmented) focal lengths (official k)."""
    return np.sqrt(100.0 / np.square(fx) + 100.0 / np.square(fy)) / C_REF


def quaternion_to_euler(q):
    """Official PrepareImageInputs_UDA.quaternion_to_euler: [pitch, yaw, roll] of a [w, x, y, z] quaternion."""
    w, x, y, z = q
    pitch = asin(2 * (w * y - x * z))
    yaw = atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    roll = atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    return np.array([pitch, yaw, roll])


def euler_to_quaternion(e, aug):
    """Official euler_to_quaternion with the perturbation (PDBEV-10: for a level camera the Euler roll tilts it and
    the Euler pitch spins it about the optical axis, so pitch_aug goes to roll and roll_aug to pitch)."""
    pitch, yaw, roll = e
    pitch += np.random.uniform(*aug['roll_aug'])
    yaw += np.random.uniform(*aug['yaw_aug'])
    roll += np.random.uniform(*aug['pitch_aug'])
    cy, sy = cos(yaw * 0.5), sin(yaw * 0.5)
    cr, sr = cos(roll * 0.5), sin(roll * 0.5)
    cp, sp = cos(pitch * 0.5), sin(pitch * 0.5)
    w = cy * cr * cp + sy * sr * sp
    x = cy * sr * cp - sy * cr * sp
    y = cy * cr * sp + sy * sr * cp
    z = sy * cr * cp - cy * sr * sp
    return np.array([w, x, y, z])


def _pose(q_wxyz, t):
    m = torch.eye(4)
    m[:3, :3] = torch.Tensor(Quaternion(*q_wxyz).rotation_matrix)
    m[:3, 3] = torch.Tensor(np.asarray(t, dtype=np.float64))
    return m


class PCCRPDBEVDataset(PCCRDetDataset):

    def __init__(self, *args, pdbev_aug=None, target_stride=8, bev_range=(-51.2, -51.2, 51.2, 51.2), **kwargs):
        super().__init__(*args, **kwargs)
        # single key frame, every camera, no LiDAR depth, no CBGS (the PCCR exps)
        assert self.num_sweeps == 1 and self.key_idxes == [0] and not self.sweeps_idx
        assert not self.return_depth and not self.use_fusion and not self.use_cbgs
        assert self.ida_aug_conf['Ncams'] == len(self.ida_aug_conf['cams'])
        assert pdbev_aug is not None or not self.is_train, 'training needs the perturbation ranges'
        self.pdbev_aug = pdbev_aug
        self.target_stride = target_stride
        self.bev_range = bev_range
        self.num_classes = len(self.classes)
        self._ida_log = []

    def sample_ida_augmentation(self):
        """NuscDetDataset's sampling (one draw per camera in get_image), recorded for the PD-BEV inputs."""
        aug = super().sample_ida_augmentation()
        self._ida_log.append(aug)
        return aug

    def __getitem__(self, idx):
        self._ida_log = []
        ret = super().__getitem__(idx)  # imgs, sensor2ego, intrins, ida, sensor2sensor, bda, ts, metas, boxes, labels
        info = self.infos[idx]
        cams = list(self.ida_aug_conf['cams'])
        assert len(self._ida_log) == len(cams)
        intrins, ida_mats, bda_mat = ret[2][0], ret[3][0], ret[5]
        actual = torch.zeros(len(cams), 4, 4)
        for k, (resize, resize_dims, crop, flip, rotate) in enumerate(self._ida_log):
            K = intrins[k, :3, :3].clone()
            K[0, :] = resize * K[0, :]
            K[1, :] = resize * K[1, :]
            K[0, 2] = K[0, 2] - crop[0]
            K[1, 2] = K[1, 2] - crop[1]
            if flip:  # PDBEV-9
                K[0, 2] = (crop[2] - crop[0]) - K[0, 2]
            actual[k, :3, :3] = K
            actual[k, 3, 3] = 1
        extra = dict(actual_intrin_mats=actual.unsqueeze(0))
        if self.is_train:
            virtual = torch.zeros(len(cams), 4, 4)
            for k, cam in enumerate(cams):
                cs = info['cam_infos'][cam]['calibrated_sensor']
                q = euler_to_quaternion(quaternion_to_euler(np.array(cs['rotation'])), self.pdbev_aug)
                t = np.asarray(cs['translation'], dtype=np.float64) + np.array([
                    np.random.uniform(*self.pdbev_aug['extri_x_aug']),
                    np.random.uniform(*self.pdbev_aug['extri_y_aug']),
                    np.random.uniform(*self.pdbev_aug['extri_z_aug'])])
                virtual[k] = _pose(q, t)
            boxes, labels = self.get_gt(info, cams)  # the 3D head's boxes in the ego frame, before the bda
            boxes, labels = self.filter_objects(boxes, labels, bda_mat)
            real_s2e = ret[1][0]
            targets = {}
            for suffix, s2e, rows, rmin, rmax in (('', real_s2e, 2, 1, 40), ('_aug', virtual, 1, 4, 200)):
                h, a, m = self.view_targets(boxes, labels, s2e, intrins, ida_mats, actual, rows, rmin, rmax)
                targets['heatmaps_2d' + suffix] = h
                targets['ann_maps_2d' + suffix] = a
                targets['heatmap_masks_2d' + suffix] = m
            extra.update(virtual_sensor2ego_mats=virtual.unsqueeze(0), targets_2d=targets)
        return ret + [extra]

    def filter_objects(self, boxes, labels, bda_mat):
        """PDBEV-4/5: labels 0..C-1 whose centre lies inside the BEV range after the bda augmentation."""
        boxes = boxes.reshape(-1, boxes.shape[-1] if boxes.numel() else 9)
        labels = labels.reshape(-1).long()
        keep = (labels >= 0) & (labels < self.num_classes)
        if len(boxes):
            xy = (boxes[:, :3].double() @ bda_mat[:3, :3].double().T)[:, :2]
            x0, y0, x1, y1 = self.bev_range
            keep &= (xy[:, 0] > x0) & (xy[:, 0] < x1) & (xy[:, 1] > y0) & (xy[:, 1] < y1)
        return boxes[keep], labels[keep]

    def view_targets(self, boxes, labels, sensor2ego, intrins, ida_mats, actual, rows_below, rmin, rmax):
        """Official get_2D_heatmap (real view: 3x3 window, radius 1..40/8) / get_2D_aug_heatmap (virtual view: 2x3
        window, radius 4..200/8) for the cameras sensor2ego (N, 4, 4) with the real cameras' intrinsics and ida."""
        fH, fW = self.ida_aug_conf['final_dim']
        s = self.target_stride
        H8, W8 = int(fH / s), int(fW / s)
        N, C = sensor2ego.shape[0], self.num_classes
        heatmaps = torch.zeros(N, H8, W8, C)
        masks = torch.zeros(N, H8, W8, C)
        anns = torch.zeros(N, H8, W8, 7, C)
        if len(boxes) == 0:
            return heatmaps, anns, masks
        lb = LiDARInstance3DBoxes(boxes[:, :7].numpy(), origin=(0.5, 0.5, 0.5))
        centers = lb.gravity_center.numpy().astype(np.float32)
        corners = lb.corners.numpy().reshape(-1, 3).astype(np.float32)
        dims = boxes[:, 3:6]
        labels = labels.numpy()
        for k in range(N):
            cam2ego = sensor2ego[k].numpy().astype(np.float32)
            K = intrins[k, :3, :3].numpy().astype(np.float32)
            uv, valid, depth = self._ego2img(centers, cam2ego, K)
            cuv, _, _ = self._ego2img(corners, cam2ego, K)
            cuv = cuv.reshape(len(boxes), 8, 2)
            box_w = cuv[:, :, 0].max(1) - cuv[:, :, 0].min(1)
            box_h = cuv[:, :, 1].max(1) - cuv[:, :, 1].min(1)
            resize = self._ida_log[k][0]
            ida_rot, ida_tran = ida_mats[k, :2, :2], ida_mats[k, :2, 3]
            uv = torch.tensor(uv).matmul(ida_rot.T) + ida_tran  # the image augmentation of this camera
            fx, fy = float(actual[k, 0, 0]), float(actual[k, 1, 1])
            depth_real = torch.tensor(depth)
            depth_virtual = torch.tensor(depth * virtual_depth_scale(fx, fy))
            uv8 = uv / s
            box_w8 = resize * box_w / s
            box_h8 = resize * box_h / s
            uvn = uv8.numpy()
            valid = valid & (uvn[:, 0] >= 0) & (uvn[:, 0] < W8) & (uvn[:, 1] >= 0) & (uvn[:, 1] < H8)  # PDBEV-6/7
            center_int = uv8.to(torch.int32)
            for j in range(len(labels)):
                if not (valid[j] and 0 <= labels[j] < C):  # PDBEV-4
                    continue
                radius = gaussian_radius((torch.tensor(box_w8[j]), torch.tensor(box_h8[j])), min_overlap=0.1)
                radius = float(radius)
                radius = int(radius) if np.isfinite(radius) else int(rmax / s)  # guard: degenerate projected box
                radius = min(max(rmin, radius), int(rmax / s))
                x_int, y_int = int(center_int[j, 0]), int(center_int[j, 1])
                lab = int(labels[j])
                draw_heatmap_gaussian(heatmaps[k, :, :, lab], [x_int, y_int], radius)
                r = slice(max(y_int - 1, 0), y_int + rows_below)
                c = slice(max(x_int - 1, 0), x_int + 2)
                masks[k, r, c, lab] = 1
                anns[k, r, c, 0, lab] = uv8[j, 0] - center_int[j, 0]
                anns[k, r, c, 1, lab] = uv8[j, 1] - center_int[j, 1]
                anns[k, r, c, 2, lab] = depth_virtual[j, 0]
                anns[k, r, c, 3:6, lab] = dims[j]
                anns[k, r, c, 6, lab] = depth_real[j, 0]
        return heatmaps, anns, masks

    @staticmethod
    def _ego2img(points_ego, cam2ego, K):
        """Official Load3DBoxesHeatmap.ego2img: pixel coordinates, depth > 1 m validity and depth."""
        pts = np.concatenate([points_ego, np.ones((points_ego.shape[0], 1), dtype=points_ego.dtype)], axis=1)
        pts_cam = (pts @ np.linalg.inv(cam2ego).T)[:, :3]
        valid = pts_cam[:, -1] > 1.0
        img = pts_cam @ K.T
        depth = img[:, 2:3].copy()
        img = img / img[:, 2:3]
        return img[:, :2], valid, depth


def collate_fn_pdbev(data):
    """PCCR collate (no depth labels) plus mats_dict['actual_intrin_mats'] and, in training,
    mats_dict['virtual_sensor2ego_mats'] (B, 1, N, 4, 4) and a 7th element: the stacked 2D targets."""
    extras = [d[10] for d in data]
    ret = pccr_collate_fn([d[:10] for d in data], is_return_depth=False)
    mats = ret[1]
    mats['actual_intrin_mats'] = torch.stack([e['actual_intrin_mats'] for e in extras])
    if 'virtual_sensor2ego_mats' in extras[0]:
        mats['virtual_sensor2ego_mats'] = torch.stack([e['virtual_sensor2ego_mats'] for e in extras])
        ret.append({k: torch.stack([e['targets_2d'][k] for e in extras]) for k in extras[0]['targets_2d']})
    return ret
