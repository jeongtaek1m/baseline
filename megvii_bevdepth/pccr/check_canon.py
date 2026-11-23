# PCCR port: new file. Checks of the canonical-camera port (as CoIn3D/pccr/check_canon_bevdepth.py does for BEVDet).
"""  python pccr/check_canon.py geometry            # CPU: band, round trip, frustum grid
     python pccr/check_canon.py data --out outputs/canon_vis   # CPU: GT boxes / depth on canonical images (R1 train,
                                                   # R1 / R1-f / R6 / R9 test) + the frustum of each box's bottom centre
     python pccr/check_canon.py bnstats            # CPU: camera-aware DepthNet BN input z-scores, plain vs canonical
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np
import torch
from pyquaternion import Quaternion

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from bevdepth.canon.geometry import canonical_cam_rotation, project_ego_to_canon, rt_to_mat  # noqa: E402
from bevdepth.datasets.pccr_canon_dataset import plain_row_band_deg  # noqa: E402
from bevdepth.exps.pccr.base_exp_pccr import build_pccr_dataset, pccr_confs, rig_dir  # noqa: E402
from bevdepth.exps.pccr.base_exp_pccr_canon import RHO, build_pccr_canon_dataset, make_canon_conf  # noqa: E402
from bevdepth.layers.backbones.canon_lss_fpn import OUTSIDE, CanonLSSFPN, canon_points  # noqa: E402

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
COLORS = [(0, 255, 0), (0, 128, 255), (255, 0, 255), (255, 255, 0), (0, 255, 255), (0, 0, 255), (255, 128, 0),
          (255, 255, 255), (128, 0, 255)]  # BGR per class


def t64(x, dtype=torch.float64):
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def bda_matrix(rng):
    a = math.radians(rng.uniform(-22.5, 22.5))
    rot = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    flip = np.diag([-1 if rng.random() < 0.5 else 1, -1 if rng.random() < 0.5 else 1, 1])
    M = np.eye(4)
    M[:3, :3] = flip @ (rng.uniform(0.95, 1.05) * np.eye(3)) @ rot
    M[:3, 3] = rng.normal(0, 0.5, 3)
    return M


class _VT:
    """CanonLSSFPN's geometry methods without building the network."""
    downsample_factor = 16
    frustum_px = 'linspace'
    depth_bins = torch.arange(2.0, 58.0, 0.5)
    pixel_grid = CanonLSSFPN.pixel_grid
    get_canon_geometry = CanonLSSFPN.get_canon_geometry


# ================================================================================================ geometry
def cmd_geometry(a):
    _, _, ida_conf, _ = pccr_confs(a.final_h)
    for band in ('plain', 'full'):
        make_canon_conf(os.path.join(a.info_root, 'R1_infos_train.pkl'), ida_conf, band=band)
    rng = np.random.default_rng(0)
    worst = {torch.float32: 0.0, torch.float64: 0.0}
    for k in range(40):
        yaw, pitch, roll = rng.uniform(-np.pi, np.pi), math.radians(rng.uniform(-20, 20)), math.radians(rng.uniform(-5, 5))
        R_src = (Quaternion(axis=[0, 0, 1], angle=yaw) * Quaternion(axis=[0, 1, 0], angle=pitch)
                 * Quaternion(axis=[1, 0, 0], angle=roll)).rotation_matrix @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        T = rt_to_mat(canonical_cam_rotation(R_src), rng.uniform([-3, -2, 0.4], [3, 2, 3.4]))
        W, H = int(rng.integers(500, 1300)), int(rng.integers(200, 500))
        intr = (RHO, RHO, W / 2 + rng.uniform(-50, 50), rng.uniform(0.2, 0.7) * H)
        ida = np.eye(4)
        if k % 2:
            ida[0, 0], ida[0, 3] = -1.0, W - 1.0
        if k % 3:
            ida[1, 3] = float(rng.integers(-20, 21))
        bda = bda_matrix(rng)
        n = 50000
        az, t, r = rng.uniform(-1.4, 1.4, n), rng.uniform(-1.5, 1.5, n), rng.uniform(1, 60, n)
        P_ego = np.stack([np.sin(az), t, np.cos(az)], -1) * r[:, None] @ T[:3, :3].T + T[:3, 3]
        u0, v0, r_p, front = project_ego_to_canon(P_ego, T, intr)     # reference numpy projection
        assert front.all()
        uv = np.stack([u0, v0], -1) @ ida[:2, :2].T + ida[:2, 3]
        ref = P_ego @ bda[:3, :3].T + bda[:3, 3]
        K = np.array([[RHO, 0, intr[2], 0], [0, RHO, intr[3], 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        for dt in worst:
            p = canon_points(t64(uv, dt)[None, None], t64(r_p, dt)[None, None], t64(T, dt)[None, None],
                             t64(K, dt)[None, None], t64(ida, dt)[None, None], t64(bda, dt)[None])[0, 0].double().numpy()
            worst[dt] = max(worst[dt], float(np.abs(p - ref).max()))
    print(f'[round trip] ego -> canonical pixel + horizontal range (reference numpy) -> ego (port torch, flip / vshift in '
          f'ida, random BDA): 40 cameras x 50k points, range 1-60 m, |az| < 80 deg, |tan el| < 1.5: max |error| '
          f'float32 {worst[torch.float32]:.2e} m, float64 {worst[torch.float64]:.2e} m -> '
          f'{"PASS" if worst[torch.float32] < 1e-3 else "FAIL"}')
    # frustum grid of CanonLSSFPN.get_canon_geometry: reprojects onto its pixels and bins; invalid cells at OUTSIDE
    vt = _VT()
    Hf, Wf, n = 16, 40, 6
    Hp, Wp = Hf * 16, Wf * 16
    Ts, Ks, idas = [], [], []
    for k in range(n):
        T = rt_to_mat(canonical_cam_rotation(Quaternion(axis=[0, 0, 1], angle=rng.uniform(-3, 3)).rotation_matrix
                                             @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])),
                      rng.uniform([-3, -2, 0.4], [3, 2, 3.4]))
        W = Wp - int(rng.integers(0, 15))
        K = np.eye(4)
        K[0, 0] = K[1, 1] = RHO
        K[0, 2], K[1, 2] = W / 2, rng.uniform(40, 80)
        ida = np.eye(4)
        if k % 2:
            ida[0, 0], ida[0, 3] = -1, W - 1
        ida[1, 3] = float(rng.integers(-20, 21))
        Ts.append(T), Ks.append(K), idas.append(ida)
    bda = bda_matrix(rng)
    valid = np.ones((1, n, Hp, Wp), np.uint8)
    valid[:, :, :, Wp - 20:] = 0
    with torch.no_grad():
        coor = vt.get_canon_geometry(t64(np.stack(Ts))[None].float(), t64(np.stack(Ks))[None].float(),
                                     t64(np.stack(idas))[None].float(), t64(bda)[None].float(), (Hf, Wf),
                                     torch.as_tensor(valid)).double().numpy()[0]
    u, v = [x.numpy() for x in vt.pixel_grid(Hf, Wf, 'cpu', torch.float64)]
    d = vt.depth_bins.double().numpy()
    ok_cols = np.round(u) < Wp - 20
    e_px, e_d, bad = 0.0, 0.0, 0
    Binv = np.linalg.inv(bda)
    for k in range(n):
        P = coor[k]
        bad += int((P[:, :, ~ok_cols] != OUTSIDE).sum())
        P = P[:, :, ok_cols].reshape(-1, 3) @ Binv[:3, :3].T + Binv[:3, 3]
        u0, v0, r, _ = project_ego_to_canon(P, Ts[k], (RHO, RHO, Ks[k][0, 2], Ks[k][1, 2]))
        uv = np.stack([u0, v0], -1) @ idas[k][:2, :2].T + idas[k][:2, 3]
        uu, vv = np.meshgrid(u[ok_cols], v)
        D = len(d)
        e_px = max(e_px, float(np.abs(uv[:, 0] - np.tile(uu.ravel(), D)).max()),
                   float(np.abs(uv[:, 1] - np.tile(vv.ravel(), D)).max()))
        e_d = max(e_d, float(np.abs(r - np.repeat(d, uu.size)).max()))
    print(f'[frustum grid] get_canon_geometry ({n} views, {Hf}x{Wf} cells, {len(d)} bins, flip / vshift, BDA): '
          f'reprojection max |du|,|dv| {e_px:.2e} px, max |range - bin| {e_d:.2e} m; invalid cells not at OUTSIDE: {bad}')


# ================================================================================================ data / figures
def box_corners(b):
    x, y, z, l, w, h, yaw = b[:7]
    c = np.array([[sx * l / 2, sy * w / 2, sz * h / 2] for sz in (-1, 1)
                  for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return c @ R.T + np.array([x, y, z])


def in_footprint(p, b, margin=0.0):
    x, y, _, l, w, _, yaw = b[:7]
    dx, dy = p[0] - x, p[1] - y
    c, s = math.cos(yaw), math.sin(yaw)
    return abs(c * dx + s * dy) <= l / 2 + margin and abs(-s * dx + c * dy) <= w / 2 + margin


def view_project(P_ego_bda, s2e, K, ida, bda):
    """Ego points after BDA -> network-image pixels and horizontal range of one canonical view."""
    Binv = np.linalg.inv(bda)
    P = P_ego_bda @ Binv[:3, :3].T + Binv[:3, 3]
    u, v, r, front = project_ego_to_canon(P, s2e, (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
    uv = np.stack([u, v], -1) @ ida[:2, :2].T + ida[:2, 3]
    return uv, r, front


def cmd_data(a):
    _, _, ida_conf, bda_conf = pccr_confs(a.final_h)
    os.makedirs(a.out, exist_ok=True)
    mean = np.array([123.675, 116.28, 103.53], np.float32)
    std = np.array([58.395, 57.12, 57.375], np.float32)
    vt = _VT()
    jobs = [('R1', 'train', i) for i in a.train_idx] + [(r, 'test', a.test_idx) for r in a.fig_rigs]
    for rig, split, idx in jobs:
        conf = make_canon_conf(os.path.join(a.info_root, 'R1_infos_train.pkl'), ida_conf, band=a.band,
                               vshift=a.vshift if split == 'train' else 0, verbose=False)
        np.random.seed(idx)
        ds = build_pccr_canon_dataset(os.path.join(a.info_root, f'{rig}_infos_{split}.pkl'), rig_dir(a.pccr_root, rig),
                                      ida_conf, bda_conf, conf, is_train=split == 'train', return_depth=True)
        item = ds[idx]
        imgs, s2e, K, ida, _, bda, _, metas, gt_boxes, gt_labels, depth, valid = item
        cams = list(ds.ida_aug_conf['cams'])
        if split == 'test':
            gt_boxes, gt_labels = ds.get_gt(ds.infos[idx], cams)
        gt_boxes = gt_boxes.numpy().reshape(-1, 9)
        gt_labels = gt_labels.numpy()
        bda_np = bda.numpy().astype(np.float64)
        Hp, Wp = imgs.shape[-2:]
        # frustum of the network for this sample (feature grid), for the nearest-frustum-point check
        with torch.no_grad():
            coor = vt.get_canon_geometry(s2e[:, :].float(), K.float(), ida.float(), bda[None].float(),
                                         (Hp // 16, Wp // 16), valid[None]).numpy()[0]   # (N, D, Hf, Wf, 3)
        u_g, v_g = [x.numpy() for x in vt.pixel_grid(Hp // 16, Wp // 16, 'cpu', torch.float64)]
        d_g = vt.depth_bins.numpy()
        tiles, e_exact, near, n_chk = [], 0.0, [], 0
        for n, cam in enumerate(cams):
            img = imgs[0, n].permute(1, 2, 0).numpy() * std + mean
            img = np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))
            m = valid[n].numpy() > 0
            img[~m] = (img[~m] * 0.35).astype(np.uint8)
            dep = depth[0, n].numpy()
            ys, xs = np.nonzero(dep)
            for y, x in zip(ys, xs):
                t = float(np.clip((dep[y, x] - 2) / 56, 0, 1))
                cv2.circle(img, (int(x), int(y)), 1, (int(255 * t), 0, int(255 * (1 - t))), -1)
            s2e_n, K_n, ida_n = (s2e[0, n].numpy().astype(np.float64), K[0, n].numpy().astype(np.float64),
                                 ida[0, n].numpy().astype(np.float64))
            s = np.linspace(0, 1, 24)
            for b, lab in zip(gt_boxes, gt_labels):
                cs = box_corners(b)
                for e0, e1 in EDGES:
                    pts = cs[e0][None] * (1 - s[:, None]) + cs[e1][None] * s[:, None]
                    uv, r, front = view_project(pts, s2e_n, K_n, ida_n, bda_np)
                    ok = front & (r > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < Wp) & (uv[:, 1] >= 0) & (uv[:, 1] < Hp)
                    for k in range(len(s) - 1):
                        if ok[k] and ok[k + 1]:
                            cv2.line(img, tuple(np.round(uv[k]).astype(int)), tuple(np.round(uv[k + 1]).astype(int)),
                                     COLORS[int(lab)], 1, cv2.LINE_AA)
                # bottom centre: exact back-projection and nearest frustum point
                bc = np.array([b[0], b[1], b[2] - b[5] / 2])
                uv, r, front = view_project(bc[None], s2e_n, K_n, ida_n, bda_np)
                ui, vi = int(np.round(uv[0, 0])), int(np.round(uv[0, 1]))
                if not (front[0] and 0 <= ui < Wp and 0 <= vi < Hp and m[vi, ui] and 2.0 <= r[0] < 58.0):
                    continue
                p = canon_points(t64(uv)[None, None], t64(r)[None, None], t64(s2e_n)[None, None], t64(K_n)[None, None],
                                 t64(ida_n)[None, None], t64(bda_np)[None])[0, 0, 0].numpy()
                e_exact = max(e_exact, float(np.abs(p - bc).max()))
                j, i = int(np.abs(u_g - uv[0, 0]).argmin()), int(np.abs(v_g - uv[0, 1]).argmin())
                dk = int(np.abs(d_g - r[0]).argmin())
                q = coor[n, dk, i, j]
                if q[0] == OUTSIDE:
                    continue
                n_chk += 1
                near.append((float(np.hypot(*(q[:2] - bc[:2]))), in_footprint(q, b), in_footprint(q, b, 0.5)))
            cv2.putText(img, cam, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
        cols = 2
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.concatenate([np.concatenate(tiles[r * cols:(r + 1) * cols], 1) for r in range(len(tiles) // cols)], 0)
        f = os.path.join(a.out, f'canon_{a.band}_{rig}_{split}_{idx}.jpg')
        cv2.imwrite(f, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
        dist = np.array([x[0] for x in near]) if near else np.zeros(1)
        print(f'{f}: {len(cams)} views {Hp}x{Wp} (canvas {metas["canon"]["hw"]}, offset {metas["canon"]["offset"]}), '
              f'cy {[round(float(k), 1) for k in K[0, :, 1, 2]]}, valid {valid.float().mean():.2f}, depth px '
              f'{(depth > 0).float().mean():.2%}; {len(gt_boxes)} GT boxes; bottom centres in view {n_chk}: exact '
              f'back-projection max err {e_exact:.1e} m, nearest frustum point dist median {np.median(dist):.2f} m '
              f'max {dist.max():.2f} m, inside footprint {np.mean([x[1] for x in near]) if near else 0:.0%} '
              f'(+0.5 m: {np.mean([x[2] for x in near]) if near else 0:.0%})', flush=True)


# ================================================================================================ BN statistics
MLP_NAMES = ['fx', 'fy', 'cx', 'cy', 'ida00', 'ida01', 'ida03', 'ida10', 'ida11', 'ida13', 'bda00', 'bda01', 'bda10',
             'bda11', 'bda22'] + [f'R{i}{j}' if j < 3 else f't{i}' for i in range(3) for j in range(4)]


def mlp_input(item, canon):
    """DepthNet.forward's 27 entries for one sample (canonical: intrinsics (rho, rho, 0, 0), as CanonLSSFPN)."""
    _, s2e, K, ida, _, bda = item[:6]
    K = K[0].clone().double()
    if canon:
        K[:, 0, 2] = 0.0
        K[:, 1, 2] = 0.0
    ida, s2e, bda = ida[0].double(), s2e[0].double(), bda.double()
    n = K.shape[0]
    rows = torch.stack([K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], ida[:, 0, 0], ida[:, 0, 1], ida[:, 0, 3],
                        ida[:, 1, 0], ida[:, 1, 1], ida[:, 1, 3]] + [bda[i, j].expand(n) for i, j in
                                                                     ((0, 0), (0, 1), (1, 0), (1, 1), (2, 2))], -1)
    return torch.cat([rows, s2e[:, :3, :].reshape(n, 12)], -1).numpy()


def cmd_bnstats(a):
    _, _, ida_conf, bda_conf = pccr_confs(a.final_h)
    conf = make_canon_conf(os.path.join(a.info_root, 'R1_infos_train.pkl'), ida_conf, band=a.band, vshift=a.vshift)
    for name, canon in (('plain row', False), (f'canonical (vshift {a.vshift})', True)):
        np.random.seed(0)

        def build(rig, split, train):
            path = os.path.join(a.info_root, f'{rig}_infos_{split}.pkl')
            if canon:
                return build_pccr_canon_dataset(path, rig_dir(a.pccr_root, rig), ida_conf, bda_conf,
                                                dict(conf, vshift=conf['vshift'] if train else 0), is_train=train,
                                                return_depth=False)
            return build_pccr_dataset(path, rig_dir(a.pccr_root, rig), ida_conf, bda_conf, is_train=train,
                                      return_depth=False)
        ds = build('R1', 'train', True)
        idx = list(range(0, len(ds), len(ds) // a.n_samples))[:a.n_samples]
        M = np.concatenate([mlp_input(ds[i], canon) for i in idx])
        mu, var = M.mean(0), M.var(0, ddof=1)
        print(f'[{name}] training statistics from {a.n_samples} R1 samples ({len(M)} views); entries with var < 1e-6: '
              f'{[k for k, v in zip(MLP_NAMES, var) if v < 1e-6]}', flush=True)
        for rig in a.rigs:
            x = mlp_input(build(rig, 'test', False)[0], canon)
            z = np.abs((x - mu) / np.sqrt(var + 1e-5))
            worst = sorted(((float(z[:, k].max()), nm) for k, nm in enumerate(MLP_NAMES)), reverse=True)[:3]
            print(f'  {rig:6s}: max |z| {z.max():9.3g}; top ' + ', '.join(f'{nm} {v:.3g}' for v, nm in worst), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['geometry', 'data', 'bnstats'])
    ap.add_argument('--info_root', default='data/pccr_megvii')
    ap.add_argument('--pccr_root', default='/data4/pccr/data')
    ap.add_argument('--final_h', type=int, default=256)
    ap.add_argument('--band', default='plain', choices=['plain', 'full'])
    ap.add_argument('--vshift', type=int, default=20)
    ap.add_argument('--out', default='outputs/canon_vis')
    ap.add_argument('--train_idx', type=int, nargs='+', default=[0, 1500])
    ap.add_argument('--test_idx', type=int, default=900)
    ap.add_argument('--fig_rigs', nargs='+', default=['R1', 'R1-f', 'R6', 'R9'])
    ap.add_argument('--rigs', nargs='+', default=['R1', 'R1-t', 'R1-r', 'R1-f', 'R1-c6', 'R1-c10', 'R2', 'R3', 'R4',
                                                  'R5', 'R6', 'R7', 'R8', 'R9'])
    ap.add_argument('--n_samples', type=int, default=40)
    a = ap.parse_args()
    {'geometry': cmd_geometry, 'data': cmd_data, 'bnstats': cmd_bnstats}[a.cmd](a)


if __name__ == '__main__':
    main()
