# PCCR port: new file, a verbatim copy of CoIn3D/BEVDet/mmdet3d/canon/geometry.py (md5 2a733e25), our canonical camera's
# numpy geometry (reference implementation); only these two comment lines are added.
"""Canonical cylindrical camera geometry (numpy), for the canonical BEVDepth plug-in.

The planning / resampling functions are copied from our BEVFormer canonical implementation
(BEVFormer_canonical/projects/mmdet3d_plugin/canonical/geometry.py, camera_models.py and the
'strip' / 'rows' planners of transforms.CanonicalCameraResample) with the same names and maths, so this
code base does not depend on that repository's path or its mmdet registry. pccr/check_canon_bevdepth.py
compares these copies with the reference implementation when it is present.

Conventions (as in the reference)
* camera frame: OpenCV (x right, y down, z forward); ego frame: x forward, y left, z up.
* canonical camera: same centre and azimuth (yaw) as the source camera, zero pitch and roll.
* cylindrical model, intr = (rho, rho, cx, cy) in pixels:
      u = rho * atan2(x, z) + cx,   v = rho * y / r + cy,   r = sqrt(x^2 + z^2)
  i.e. a pixel (u, v) has azimuth a = (u - cx) / rho and tan(elevation, down positive) t = (v - cy) / rho, and
  its ray is (sin a, t, cos a): the point at horizontal range r along it is r * (sin a, t, cos a).
"""
import math

import numpy as np

CYLINDRICAL = 'cylindrical'


# ---------------------------------------------------------------------------- reference copies
def canonical_cam_rotation(R_cam2ego):
    """Levelled camera -> ego rotation: the source optical axis projected on the ground plane, image x parallel to
    the ground, image y straight down (reference geometry.canonical_cam_rotation with R_ego2ground = I)."""
    R = np.asarray(R_cam2ego, dtype=np.float64)
    axis = R[:, 2]
    yaw = math.atan2(axis[1], axis[0])
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    return np.stack([right, down, fwd], axis=1)


def rt_to_mat(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def unproject_cyl_np(u, v, intr):
    """Pixel -> unit ray in the canonical camera frame (reference camera_models.unproject_np, cylindrical)."""
    fx, fy, cx, cy = intr
    az = (u - cx) / fx
    d = np.stack([np.sin(az), (v - cy) / fy, np.cos(az)], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def resample_maps_cyl(K_src, R_src2ego, src_hw, R_can2ego, intr_can, can_hw, row_col_offset=(0, 0)):
    """cv2.remap maps from a pinhole source to the canonical cylinder and the valid mask
    (reference geometry.resample_maps_model with model='cylindrical')."""
    h_c, w_c = can_hw
    h_s, w_s = src_hw
    r0, c0 = row_col_offset
    u, v = np.meshgrid(np.arange(c0, c0 + w_c, dtype=np.float64), np.arange(r0, r0 + h_c, dtype=np.float64))
    d_can = unproject_cyl_np(u, v, intr_can)
    d_src = d_can @ (R_src2ego.T @ R_can2ego).T
    pix = d_src @ np.asarray(K_src, dtype=np.float64).T
    z = pix[..., 2]
    in_front = z > 1e-6
    zs = np.where(in_front, z, 1.0)
    xs = pix[..., 0] / zs
    ys = pix[..., 1] / zs
    valid = in_front & (xs >= 0) & (xs <= w_s - 1) & (ys >= 0) & (ys <= h_s - 1)
    map_x = np.where(valid, xs, -1.0).astype(np.float32)
    map_y = np.where(valid, ys, -1.0).astype(np.float32)
    return map_x, map_y, valid


def _border_rays(K_src, R_src2ego, src_hw, R_can2ego, n=64):
    h, w = src_hw
    t = np.linspace(0, 1, n)
    border = np.concatenate([
        np.stack([t * (w - 1), np.zeros(n)], 1),
        np.stack([t * (w - 1), np.full(n, h - 1.0)], 1),
        np.stack([np.zeros(n), t * (h - 1)], 1),
        np.stack([np.full(n, w - 1.0), t * (h - 1)], 1)])
    rays = np.concatenate([border, np.ones((len(border), 1))], 1) @ np.linalg.inv(K_src).T
    return rays @ (R_can2ego.T @ R_src2ego).T


def source_half_azimuth(K_src, R_src2ego, src_hw, R_can2ego, n=64):
    """Max |azimuth| (rad) of the source image border seen from the levelled camera."""
    rays = _border_rays(K_src, R_src2ego, src_hw, R_can2ego, n)
    return float(np.max(np.abs(np.arctan2(rays[:, 0], rays[:, 2]))))


def elevation_cover_tan(K_src, R_src2ego, src_hw, R_can2ego, n=64):
    """[min, max] of tan(elevation) (positive up) of the source image seen from the levelled camera."""
    rays = _border_rays(K_src, R_src2ego, src_hw, R_can2ego, n)
    tan_el = -rays[:, 1] / np.maximum(np.hypot(rays[:, 0], rays[:, 2]), 1e-9)
    return float(tan_el.min()), float(tan_el.max())


def cylinder_layout(rho, elev_range_deg, half_az):
    """Strip layout: columns span azimuth [-half_az, half_az], rows elevation [lo, hi] (deg, up positive)."""
    lo, hi = [math.radians(e) for e in elev_range_deg]
    W = 2 * int(math.ceil(rho * half_az))
    H = int(math.ceil(rho * (math.tan(hi) - math.tan(lo))))
    return H, W, (rho, rho, W / 2.0, rho * math.tan(hi))


def valid_bbox(valid_masks):
    union = np.any(np.stack(valid_masks), axis=0)
    rows = np.where(union.any(1))[0]
    cols = np.where(union.any(0))[0]
    if len(rows) == 0:
        return 0, union.shape[0], 0, union.shape[1]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


# ------------------------------------------------------------------------- projection helpers
def project_cyl_np(P_cam, intr):
    """Canonical-camera-frame points (..., 3) -> u, v, r (horizontal range), in_front (z > 0)."""
    fx, fy, cx, cy = intr
    x, y, z = P_cam[..., 0], P_cam[..., 1], P_cam[..., 2]
    r = np.sqrt(x ** 2 + z ** 2)
    return fx * np.arctan2(x, z) + cx, fy * y / np.maximum(r, 1e-9) + cy, r, z > 1e-6


def project_ego_to_canon(P_ego, T_can2ego, intr):
    """Ego points (M, 3) -> u, v, r, in_front in one canonical view (pixel coordinates of the cropped canonical
    image before flip / padding)."""
    R, t = T_can2ego[:3, :3], T_can2ego[:3, 3]
    P_cam = (np.asarray(P_ego, dtype=np.float64) - t) @ R
    return project_cyl_np(P_cam, intr)
