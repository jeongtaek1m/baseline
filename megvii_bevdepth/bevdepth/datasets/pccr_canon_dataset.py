# PCCR port: new file. Our canonical-camera inputs on the Megvii dataset (port of CoIn3D/BEVDet/mmdet3d/canon/pipeline.py).
"""Every camera of a sample is resampled to the levelled cylindrical camera of fixed angular resolution rho (layout
'strip': one view per camera, the rows covering an elevation band, all views as wide as the widest camera, cropped to
the union of valid pixels, zero-padded to a multiple of 32), with the reference geometry (bevdepth/canon/geometry.py,
a verbatim copy). The dataset returns the official Megvii tuple with the canonical parameters in it:

- intrin_mats: K = [[rho, 0, cx], [0, rho, cy], [0, 0, 1]] of the cropped, unflipped canonical image (cylindrical:
  u = rho * atan2(x, z) + cx, v = rho * y / r + cy, r = sqrt(x^2 + z^2));
- sensor2ego_mats: the levelled canonical camera -> key ego (same centre and yaw as the camera, pitch = roll = 0),
  through the official chain global2keyego @ ego2global @ cam2ego;
- ida_mats: only the per-view horizontal flip (u' = W - 1 - u, p = 0.5, train) and the optional vertical shift
  (v' = v + dv, dv ~ U{-vshift..vshift}, zero fill, train). The official resize / crop / rotate are dropped: a resize
  changes rho, a crop moves the horizon row and the azimuth origin, an in-plane rotation is a roll;
- depth labels: the LiDAR's horizontal range along the canonical ray (z-buffer: nearest per pixel, valid pixels
  only; reference depth_map), so the official get_downsampled_gt_depth bins the horizontal range;
- the valid mask (N, Hp, Wp) as an extra last element (collate_fn below puts it in mats_dict['canon_valid']).
GT boxes, BDA and img_metas are the official ones (ego frame of the key sample), so the evaluation bridge is unchanged.

Elevation band (rule of this port): the band the plain row sees of R1. Megvii's official test crop keeps the bottom
final_dim[0] rows of the image resized to width 704; plain_row_band_deg computes, from R1's own calibration, the
elevation of the top edge of the kept rows and of R1's lower image edge at the principal column (final_dim (256, 704):
-19.71 .. +6.00 deg), and the strip keeps round(rho * (tan hi - tan lo)) rows (= 256: the reference's ceil gives 257
because rho = 552.75 is 0.04 % above R1's focal at width 704; the 257th row lies inside R1's lower edge by 0.1 px only,
is invalid on R1, and dropping it keeps every rig on R1's padded canvas height). band='full' gives R1's whole
vertical field of view (+-19.71 deg, the reference's +-19.72).

Anti-aliasing (prefilter, a choice of this port): the reference read 800 x 450 images (PIL-downsampled with
antialiasing), Megvii reads the raw 1280 x 720 photos; to give the canonical model the same filtered input as the
plain row (PIL resize to 0.55), each photo is first PIL-resized by s = min(1, rho / fx) (K scaled with it; exact), so
the cylindrical remap that follows is close to 1:1.
"""
import math
import os

import cv2
import mmcv
import numpy as np
import torch
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from PIL import Image
from pyquaternion import Quaternion

from bevdepth.canon.geometry import (canonical_cam_rotation, cylinder_layout, project_ego_to_canon, resample_maps_cyl,
                                     rt_to_mat, source_half_azimuth, valid_bbox)
from bevdepth.datasets.nusc_det_dataset import bev_transform
from bevdepth.datasets.nusc_det_dataset import collate_fn as official_collate_fn
from bevdepth.datasets.pccr_det_dataset import PCCRDetDataset

__all__ = ['PCCRCanonDetDataset', 'canon_collate_fn', 'plain_row_band_deg', 'DEFAULT_CANON']

DEFAULT_CANON = dict(
    rho=552.75,               # px/rad: R1's 1004.6 px/rad x 0.55 (the official 704 / 1280), as the reference
    elev_range_deg=None,      # (lo, hi) deg, up positive; set by the exp from R1 (plain_row_band_deg)
    max_rows=None,            # strip height cap (round(rho * dtan) of the band); None: the reference's ceil
    max_half_az_deg=89.0,
    crop_to_valid=True,
    level_pitch_roll=True,
    pad_divisor=32,           # reference 16 (BEVDet's stride-16 grid); Megvii's SECONDFPN concatenates the stride-4..32
                              # maps, so H and W must be multiples of 32 (as the official 256 x 704); R1: 256 x 626 -> 640
    vshift=0,                 # training only
    prefilter=True,
    interpolation='linear',
    min_depth=0.1,            # reference depth_map limits
    max_depth=200.0,
    cache_size=512,
)


def _T(translation, rotation_wxyz):
    T = np.eye(4)
    T[:3, :3] = Quaternion(*rotation_wxyz).rotation_matrix
    T[:3, 3] = translation
    return T


def plain_row_band_deg(cam_infos, final_dim, bot_pct_lim=(0.0, 0.0), full=False):
    """Elevation band (deg, up positive) of the levelled cameras of one (training) sample that the plain row's
    official test-time resize + crop keeps (NuscDetDataset.sample_ida_augmentation, is_train=False): top edge of the
    kept rows and bottom image edge, at the principal column, in K's pixel-edge convention (cx = W / 2).
    full=True: the whole image (top edge 0)."""
    fH, fW = final_dim
    los, his = [], []
    for c in cam_infos.values():
        K = np.asarray(c['calibrated_sensor']['camera_intrinsic'], dtype=np.float64)
        R = Quaternion(c['calibrated_sensor']['rotation']).rotation_matrix
        H, W = int(c['height']), int(c['width'])
        resize = max(fH / H, fW / W)
        newH = int(H * resize)
        crop_h = int((1 - np.mean(bot_pct_lim)) * newH) - fH
        v_top = 0.0 if full else max(crop_h, 0) / resize
        v_bot = float(H) if full else min(float(H), (crop_h + fH) / resize)
        M = canonical_cam_rotation(R).T @ R @ np.linalg.inv(K)

        def tan_el(v):
            r = M @ np.array([K[0, 2], v, 1.0])
            return -r[1] / math.hypot(r[0], r[2])

        his.append(tan_el(v_top))
        los.append(tan_el(v_bot))
    return math.degrees(math.atan(min(los))), math.degrees(math.atan(max(his)))


class PCCRCanonDetDataset(PCCRDetDataset):

    def __init__(self, *args, canon_conf=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.key_idxes == [0] and self.num_sweeps == 1 and not self.use_fusion, 'canonical: single frame'
        c = dict(DEFAULT_CANON, **(canon_conf or {}))
        assert c['elev_range_deg'] is not None, 'set canon_conf elev_range_deg (plain_row_band_deg)'
        self.canon = c
        self.rho = float(c['rho'])
        self.max_half_az = math.radians(c['max_half_az_deg'])
        self.interp = dict(linear=cv2.INTER_LINEAR, nearest=cv2.INTER_NEAREST, cubic=cv2.INTER_CUBIC)[c['interpolation']]
        self._cache = {}

    # ------------------------------------------------------------------ calibration-only quantities (cached)
    @staticmethod
    def _r(a, n):
        return tuple(np.round(np.asarray(a, dtype=np.float64), n).ravel())

    def _cached(self, key, fn):
        if key not in self._cache:
            if len(self._cache) >= self.canon['cache_size']:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = fn()
        return self._cache[key]

    def _maps(self, K, R, src_hw, Rc, intr, can_hw, offset=(0, 0)):
        key = ('maps', self._r(K, 3), self._r(R, 6), tuple(src_hw), self._r(Rc, 6), self._r(intr, 3),
               tuple(can_hw), tuple(offset))
        return self._cached(key, lambda: resample_maps_cyl(K, R, src_hw, Rc, intr, can_hw, offset))

    def _rotation(self, R):
        return canonical_cam_rotation(R) if self.canon['level_pitch_roll'] else np.asarray(R, dtype=np.float64)

    def _plan_strip(self, Ks, Rs, src_hws):
        """Reference PrepareCanonicalInputs._plan_strip, plus the row cap max_rows (bottom rows dropped, cy kept)."""
        Rcs = [self._rotation(R) for R in Rs]
        half = max(min(source_half_azimuth(K, R, hw, Rc), self.max_half_az)
                   for K, R, hw, Rc in zip(Ks, Rs, src_hws, Rcs))
        H, W, intr = cylinder_layout(self.rho, self.canon['elev_range_deg'], half)
        if self.canon['max_rows']:
            H = min(H, int(self.canon['max_rows']))
        plans = [dict(src=i, Rc=Rc, intr=intr) for i, Rc in enumerate(Rcs)]
        offset = (0, 0)
        if self.canon['crop_to_valid']:
            valids = [self._maps(K, R, hw, p['Rc'], p['intr'], (H, W))[2] for K, R, hw, p in zip(Ks, Rs, src_hws, plans)]
            r0, r1, c0, c1 = valid_bbox(valids)
            offset, (H, W) = (int(r0), int(c0)), (int(r1 - r0), int(c1 - c0))
        return plans, (H, W), offset

    def plan(self, Ks, Rs, src_hws):
        key = ('plan', tuple(self._r(K, 3) for K in Ks), tuple(self._r(R, 6) for R in Rs),
               tuple(tuple(hw) for hw in src_hws))
        return self._cached(key, lambda: self._plan_strip(Ks, Rs, src_hws))

    # ------------------------------------------------------------------ depth GT (reference depth_map)
    def depth_map(self, P_ego, T_can2ego, intr, valid):
        H, W = valid.shape
        u, v, r, front = project_ego_to_canon(P_ego, T_can2ego, intr)
        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        keep = (front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (r > self.canon['min_depth'])
                & (r < self.canon['max_depth']))
        ui, vi, r = ui[keep], vi[keep], r[keep]
        keep = valid[vi, ui]
        ui, vi, r = ui[keep], vi[keep], r[keep]
        ranks = vi * W + ui
        order = np.lexsort((r, ranks))
        ranks, ui, vi, r = ranks[order], ui[order], vi[order], r[order]
        first = np.ones(len(ranks), dtype=bool)
        first[1:] = ranks[1:] != ranks[:-1]
        depth = np.zeros((H, W), dtype=np.float32)
        depth[vi[first], ui[first]] = r[first]
        return depth

    @staticmethod
    def _vshift(a, dv):
        out = np.zeros_like(a)
        if dv > 0:
            out[dv:] = a[:-dv]
        else:
            out[:dv] = a[-dv:]
        return out

    @staticmethod
    def _pad(a, Hp, Wp):
        pad = [(0, Hp - a.shape[0]), (0, Wp - a.shape[1])] + [(0, 0)] * (a.ndim - 2)
        return np.pad(a, pad, mode='constant', constant_values=0)

    # ------------------------------------------------------------------ images (replaces NuscDetDataset.get_image)
    def get_canon_image(self, key_info, cams, lidar_info=None):
        srcs, Ks, T_c2es, T_e2gs = [], [], [], []
        for cam in cams:
            c = key_info[cam]
            img = Image.open(os.path.join(self.data_root, c['filename']))  # as the official get_image (RGB)
            K = np.asarray(c['calibrated_sensor']['camera_intrinsic'], dtype=np.float64)[:3, :3].copy()
            if self.canon['prefilter']:
                s = min(1.0, self.rho / max(K[0, 0], K[1, 1]))
                if s < 1.0:
                    w, h = img.size
                    size = (int(round(w * s)), int(round(h * s)))
                    img = img.resize(size)  # the official resize call (same PIL filter as the plain row)
                    K[0] *= size[0] / w
                    K[1] *= size[1] / h
            srcs.append(np.ascontiguousarray(np.array(img)))
            Ks.append(K)
            T_c2es.append(_T(c['calibrated_sensor']['translation'], c['calibrated_sensor']['rotation']))
            T_e2gs.append(_T(c['ego_pose']['translation'], c['ego_pose']['rotation']))
        src_hws = [im.shape[:2] for im in srcs]
        plans, (H, W), offset = self.plan(Ks, [T[:3, :3] for T in T_c2es], src_hws)
        d = self.canon['pad_divisor']
        Hp, Wp = int(math.ceil(H / d) * d), int(math.ceil(W / d) * d)
        T_keye2g = T_e2gs[0]  # key camera ego pose (all cameras share it in PCCR)
        P_lidar = None
        if self.return_depth:
            P_lidar = np.fromfile(os.path.join(self.data_root, lidar_info['LIDAR_TOP']['filename']), dtype=np.float32,
                                  count=-1).reshape(-1, 5)[..., :3].astype(np.float64)
            T_l2g = (_T(lidar_info['LIDAR_TOP']['ego_pose']['translation'], lidar_info['LIDAR_TOP']['ego_pose']['rotation'])
                     @ _T(lidar_info['LIDAR_TOP']['calibrated_sensor']['translation'],
                          lidar_info['LIDAR_TOP']['calibrated_sensor']['rotation']))

        imgs, s2es, intrins, idas, valids, depths = [], [], [], [], [], []
        for p in plans:
            i = p['src']
            map_x, map_y, valid = self._maps(Ks[i], T_c2es[i][:3, :3], src_hws[i], p['Rc'], p['intr'], (H, W), offset)
            can = cv2.remap(srcs[i], map_x, map_y, self.interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            rho, _, cx, cy = p['intr']
            intr = (rho, rho, cx - offset[1], cy - offset[0])                  # cropped canonical image
            T_can2camego = rt_to_mat(p['Rc'], T_c2es[i][:3, 3])
            depth = None
            if self.return_depth:                                              # LiDAR -> this camera's ego
                T = np.linalg.inv(T_e2gs[i]) @ T_l2g
                depth = self.depth_map(P_lidar @ T[:3, :3].T + T[:3, 3], T_can2camego, intr, valid)
            ida = np.eye(4)
            if self.is_train and self.ida_aug_conf['rand_flip'] and np.random.choice([0, 1]):   # u' = W - 1 - u
                can, valid = can[:, ::-1], valid[:, ::-1]
                depth = depth[:, ::-1] if depth is not None else None
                ida[0, 0], ida[0, 3] = -1.0, W - 1.0
            if self.is_train and self.canon['vshift'] > 0:                     # v' = v + dv
                dv = int(np.random.randint(-self.canon['vshift'], self.canon['vshift'] + 1))
                if dv != 0:
                    can, valid = self._vshift(can, dv), self._vshift(valid, dv)
                    depth = self._vshift(depth, dv) if depth is not None else None
                    ida[1, 3] = float(dv)
            can = self._pad(np.ascontiguousarray(can), Hp, Wp)
            img = mmcv.imnormalize(can, self.img_mean, self.img_std, self.to_rgb)   # official normalisation
            imgs.append(torch.from_numpy(img).permute(2, 0, 1))
            valids.append(torch.from_numpy(self._pad(np.ascontiguousarray(valid).astype(np.uint8), Hp, Wp)))
            if depth is not None:
                depths.append(torch.from_numpy(self._pad(np.ascontiguousarray(depth), Hp, Wp)))
            K4 = np.eye(4)
            K4[0, 0], K4[1, 1], K4[0, 2], K4[1, 2] = intr[0], intr[1], intr[2], intr[3]
            s2es.append(torch.Tensor(np.linalg.inv(T_keye2g) @ T_e2gs[i] @ T_can2camego))  # official chain
            intrins.append(torch.Tensor(K4))
            idas.append(torch.Tensor(ida))
        n = len(plans)
        ego2global_rotation = np.mean([key_info[cam]['ego_pose']['rotation'] for cam in cams], 0)
        ego2global_translation = np.mean([key_info[cam]['ego_pose']['translation'] for cam in cams], 0)
        img_metas = dict(box_type_3d=LiDARInstance3DBoxes, ego2global_translation=ego2global_translation,
                         ego2global_rotation=ego2global_rotation,
                         canon=dict(hw=(H, W), pad_hw=(Hp, Wp), offset=offset))
        ret = [torch.stack(imgs)[None], torch.stack(s2es)[None], torch.stack(intrins)[None], torch.stack(idas)[None],
               torch.eye(4).expand(1, n, 4, 4).clone(),
               torch.tensor([key_info[cam]['timestamp'] for cam in cams])[None], img_metas, torch.stack(valids)]
        if self.return_depth:
            ret.append(torch.stack(depths)[None])
        return ret

    def __getitem__(self, idx):
        """NuscDetDataset.__getitem__ for a single frame with the canonical images; the valid mask is appended."""
        if self.use_cbgs:
            idx = self.sample_indices[idx]
        cams = self.choose_cams()
        info = self.infos[idx]
        data = self.get_canon_image(info['cam_infos'], cams, info['lidar_infos'])
        (sweep_imgs, sweep_sensor2ego_mats, sweep_intrins, sweep_ida_mats, sweep_sensor2sensor_mats, sweep_timestamps,
         img_metas, valid) = data[:8]
        img_metas['token'] = info['sample_token']
        if self.is_train:
            gt_boxes, gt_labels = self.get_gt(info, cams)
        else:
            gt_boxes = sweep_imgs.new_zeros(0, 7)
            gt_labels = sweep_imgs.new_zeros(0, )
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()
        bda_mat = sweep_imgs.new_zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = bev_transform(gt_boxes, rotate_bda, scale_bda, flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        ret_list = [sweep_imgs, sweep_sensor2ego_mats, sweep_intrins, sweep_ida_mats, sweep_sensor2sensor_mats,
                    bda_mat, sweep_timestamps, img_metas, gt_boxes, gt_labels]
        if self.return_depth:
            ret_list.append(data[8])
        ret_list.append(valid)
        return ret_list


def canon_collate_fn(data, is_return_depth=False):
    """The official collate_fn; the valid masks (last element) go to mats_dict['canon_valid'] (B, N, Hp, Wp)."""
    ret = official_collate_fn([d[:-1] for d in data], is_return_depth=is_return_depth)
    ret[1]['canon_valid'] = torch.stack([d[-1] for d in data])
    return ret
