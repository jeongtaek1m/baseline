"""CPU checks for the UDGA port on the co-author's Megvii PCCR base (gen2e-megvii env).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 ~/miniconda3/envs/gen2e-megvii/bin/python -I tools/check_udga_megvii.py \
        [--only NAME ...]

Each check measures the property against an independent computation and prints PASS / FAIL:
  test-inputs  the UDGA test dataset returns exactly the co-author's PCCRDetDataset images and matrices (no ext aug)
  ext-aug      about half of the training samples get +{-90, 90, 180} deg on the bda rotation, and the boxes follow
               bda_mat (boxes and camera geometry stay consistent)
  corr         every checked L_ov correspondence: the source cell's point (recomputed in float64 from the pixel grid,
               the virtual depth, k and the cameras) lands in the target cell's pixel and virtual-depth bin
               (UDGA-1/3/7); number of overlapping pairs on R1
  ld-target    UDGA-Ld: the LiDAR depth target d * k, labelled by the base's get_downsampled_gt_depth, maps to a bin
               whose lifted point (get_geometry, camera-frame depth) is within one bin of the patch's LiDAR depth
  lp-identity  UDGA-Lp: warping a camera onto itself gives zero photometric error
  lp-depth     UDGA-Lp: over the overlap pixels the photometric error is smallest at the LiDAR depth (filled) among
               scaled depths (x0.7 .. x1.3) and a constant 10 m; test-time and augmented training samples
  lp-grad      UDGA-Lp: photometric_loss_paper gives the depth logits a finite, non-zero gradient
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MEGVII = os.path.join(REPO, 'megvii_bevdepth')
sys.path.insert(0, MEGVII)
os.chdir(MEGVII)

import mmcv  # noqa: E402

from bevdepth.datasets.pccr_det_dataset import PCCR_CLASSES, collate_fn  # noqa: E402
from bevdepth.datasets.pccr_udga_dataset import PCCRUDGADataset  # noqa: E402
from bevdepth.exps.nuscenes.base_exp import img_conf  # noqa: E402
from bevdepth.exps.pccr.base_exp_pccr import build_pccr_dataset, ida_conf_for, pccr_confs, rig_dir  # noqa: E402
from bevdepth.models import udga_bev_depth as U  # noqa: E402

PCCR_ROOT = os.path.join(REPO, 'data/pccr')
D_BOUND = [1.0, 90.0, 1.0]
RESULTS = {}


def report(name, ok, msg):
    RESULTS[name] = ok
    print(f'[{"PASS" if ok else "FAIL"}] {name}: {msg}', flush=True)


def make_ds(rig, split, is_train, return_depth=False, ext_aug_prob=0.5):
    _, _, ida, bda = pccr_confs(256)
    info = f'data/pccr_megvii/{rig}_infos_{split}.pkl'
    conf = ida_conf_for(ida, mmcv.load(info))
    return PCCRUDGADataset(ida_aug_conf=conf, bda_aug_conf=bda, classes=PCCR_CLASSES,
                           data_root=rig_dir(PCCR_ROOT, rig), info_paths=info, is_train=is_train,
                           return_depth=return_depth, img_conf=img_conf, ext_aug_prob=ext_aug_prob)


_BB = {}


def backbone():
    if 'bb' not in _BB:
        from bevdepth.layers.backbones.udga_lss_fpn import UDGALSSFPN
        bconf, _, _, _ = pccr_confs(256)
        bconf['d_bound'] = list(D_BOUND)
        bconf['img_backbone_conf'] = dict(bconf['img_backbone_conf'], init_cfg=None)
        _BB['bb'] = UDGALSSFPN(pixel_const=400.0, **bconf).eval()
    return _BB['bb']


def batch_of(ds, idx, depth):
    return collate_fn([ds[idx]], is_return_depth=depth)


def check_test_inputs(args):
    bad = []
    for rig in ('R1', 'R6'):
        _, _, ida, bda = pccr_confs(256)
        info = f'data/pccr_megvii/{rig}_infos_test.pkl'
        ref = build_pccr_dataset(info, rig_dir(PCCR_ROOT, rig), ida, bda, is_train=False, return_depth=False,
                                 img_conf=img_conf)
        ds = make_ds(rig, 'test', False, ext_aug_prob=0.0)
        for i in (0, 777):
            a, b = ref[i], ds[i]
            for j in (0, 1, 2, 3, 4, 5):
                if not torch.equal(a[j], b[j]):
                    bad.append(f'{rig}#{i} item {j}')
            if a[7]['token'] != b[7]['token']:
                bad.append(f'{rig}#{i} token')
    report('test-inputs', not bad, 'identical images / matrices / tokens on R1, R6' if not bad else str(bad))


def check_ext_aug(args):
    ds = make_ds('R1', 'train', True)
    lim = ds.bda_aug_conf['rot_lim'][1]
    angles = np.array([ds.sample_bda_augmentation()[0] for _ in range(4000)])
    ext = np.abs(angles) > lim + 1e-6
    base = angles.copy()
    for a in ds.ext_aug_angles:  # remove the ext angle where it explains the value
        m = ext & (np.abs(angles - a) <= lim + 1e-6)
        base[m] -= a
    ext_ok = bool(np.all(np.abs(base) <= lim + 1e-6))
    frac = ext.mean()
    ds.ext_aug_prob = 1.0  # every sample rotated: the boxes must follow bda_mat
    err = 0.0
    for i in (3, 300, 1500):
        ret = ds[i]
        boxes, _ = ds.get_gt(ds.infos[i], list(ds.ida_aug_conf['cams']))
        if len(boxes):
            want = boxes[:, :3].double() @ ret[5][:3, :3].double().T
            err = max(err, float((ret[8][:, :3].double() - want).abs().max()))
    ok = abs(frac - 0.5) < 0.04 and ext_ok and err < 1e-3
    report('ext-aug', ok, f'{100 * frac:.1f}% of 4000 draws rotated by an ext angle (want 50%), every angle = '
           f'ext angle + U(-{lim}, {lim}): {ext_ok}; boxes vs bda_mat max err {err:.2e} m')


def _k_np(K, ida):
    A = ida[:3, :3] @ K[:3, :3]
    return 400.0 * np.sqrt(1.0 / A[0, 0]**2 + 1.0 / A[1, 1]**2)


def check_corr(args):
    ds = make_ds('R1', 'train', True)
    np.random.seed(1)
    imgs, mats = batch_of(ds, 11, False)[:2]
    bb = backbone()
    corr = U.frustum_correspondences(bb, mats)
    D, fH, fW, _ = bb.frustum.shape
    W1, H1 = float(bb.frustum[0, 0, -1, 0]), float(bb.frustum[0, -1, 0, 1])
    d0, _, step = D_BOUND
    s2e = mats['sensor2ego_mats'][0, 0].double().numpy()
    K = mats['intrin_mats'][0, 0].double().numpy()
    ida = mats['ida_mats'][0, 0].double().numpy()
    N = s2e.shape[0]
    P = corr['src'].numel()
    sel = np.random.RandomState(0).choice(P, min(P, 30000), replace=False)
    src = corr['src'][sel].numpy()
    dst = corr['dst'][sel].numpy()
    i, rem = np.divmod(src, D * fH * fW)
    d, rem = np.divmod(rem, fH * fW)
    h, w = np.divmod(rem, fW)
    j, rem2 = np.divmod(dst, D * fH * fW)
    d2, rem2 = np.divmod(rem2, fH * fW)
    h2, w2 = np.divmod(rem2, fW)
    bad, amb = 0, 0
    for n in range(len(sel)):
        a, b = i[n], j[n]
        x, y = w[n] * W1 / (fW - 1), h[n] * H1 / (fH - 1)
        z = (d0 + d[n] * step) / _k_np(K[a], ida[a])
        u0 = np.linalg.solve(ida[a][:2, :2], np.array([x, y]) - ida[a][:2, 3])
        cam = z * np.linalg.solve(K[a][:3, :3], np.array([u0[0], u0[1], 1.0]))
        ego = s2e[a][:3, :3] @ cam + s2e[a][:3, 3]
        cj = s2e[b][:3, :3].T @ (ego - s2e[b][:3, 3])
        pj = K[b][:3, :3] @ cj
        uv = ida[b][:2, :2] @ (pj[:2] / pj[2]) + ida[b][:2, 3]
        cf, rf = uv[0] * (fW - 1) / W1, uv[1] * (fH - 1) / H1
        vb = (cj[2] * _k_np(K[b], ida[b]) - d0) / step
        if (abs(cf - np.floor(cf) - 0.5) < 1e-4 or abs(rf - np.floor(rf) - 0.5) < 1e-4
                or abs(vb - np.round(vb)) < 1e-4):
            amb += 1
            continue
        if not (int(np.round(cf)) == w2[n] and int(np.round(rf)) == h2[n] and int(np.floor(vb)) == d2[n] and a != b):
            bad += 1
    pairs = corr['pairs'][0]
    ok = P > 0 and bad == 0
    report('corr', ok, f'{P} correspondences, {len(pairs)} ordered pairs on R1 ({N} cameras); checked {len(sel)}: '
           f'wrong target cell {bad}, rounding ties skipped {amb}')


def check_ld_target(args):
    from bevdepth.exps.nuscenes.base_exp import BEVDepthLightningModel
    ds = make_ds('R1', 'train', True, return_depth=True)
    np.random.seed(2)
    imgs, mats, _, _, _, _, depth = batch_of(ds, 21, True)
    bb = backbone()
    gt = depth[:, 0]  # (1, N, H, W) metric LiDAR depth
    k = bb.depth_scale(mats['intrin_mats'][:, 0], mats['ida_mats'][:, 0])  # (1, N)

    class _S:
        downsample_factor, dbound, depth_channels = 16, D_BOUND, 89
    lab = BEVDepthLightningModel.get_downsampled_gt_depth(_S, gt * k[..., None, None])  # (N*h*w, D)
    B, N, H, W = gt.shape
    h, w = H // 16, W // 16
    patches = gt.view(N, h, 16, w, 16).permute(0, 1, 3, 2, 4).reshape(-1, 256)
    dmin = torch.where(patches > 0, patches, torch.full_like(patches, 1e5)).min(1).values
    fg = lab.max(1).values > 0
    b = lab.argmax(1)
    geom = bb.get_geometry(mats['sensor2ego_mats'][:, 0], mats['intrin_mats'][:, 0], mats['ida_mats'][:, 0], None)[0]
    s2e = mats['sensor2ego_mats'][0, 0]
    errs = []
    cam_idx = torch.arange(N).view(N, 1, 1).expand(N, h, w).reshape(-1)
    hh = torch.arange(h).view(1, h, 1).expand(N, h, w).reshape(-1)
    ww = torch.arange(w).view(1, 1, w).expand(N, h, w).reshape(-1)
    for c, bi, r, q, dm in zip(cam_idx[fg], b[fg], hh[fg], ww[fg], dmin[fg]):
        p = geom[c, bi, r, q]
        zc = (torch.inverse(s2e[c]) @ torch.cat([p, p.new_ones(1)]))[2]
        errs.append(float((zc - dm).abs() * k[0, c] / D_BOUND[2]))  # in bins
    worst = max(errs) if errs else float('inf')
    ok = len(errs) > 0 and worst < 1.0 + 1e-3
    report('ld-target', ok, f'{len(errs)} labelled cells, k per camera {k[0].numpy().round(3)}: '
           f'|lifted depth of the label bin - LiDAR depth| <= {worst:.3f} bin (must be < 1)')


def _unit(imgs):
    return U.to_unit_rgb(imgs)


def check_lp_identity(args):
    ds = make_ds('R1', 'train', True)
    np.random.seed(3)
    imgs, mats = batch_of(ds, 31, False)[:2]
    rgb = _unit(imgs[:, 0])
    depth = torch.full(rgb.shape[:2] + rgb.shape[-2:], 15.0)
    ssim = U.SSIM()
    maps = U.photometric_error_maps(depth, rgb, *U.bevdet_cams(mats), [[(n, n) for n in range(rgb.shape[1])]], ssim)
    worst = max(pe[v].abs().max().item() for pe, v in maps)
    frac = np.mean([v.float().mean().item() for _, v in maps])
    report('lp-identity', worst < 1e-3, f'self-warp error max {worst:.2e} over valid pixels (valid fraction '
           f'{frac:.3f}, augmentation padding excluded)')


def _dense(depth):
    from scipy import ndimage
    d = depth.numpy()
    m = d > 0
    idx = ndimage.distance_transform_edt(~m, return_distances=False, return_indices=True)
    return torch.from_numpy(d[tuple(idx)]), torch.from_numpy(m)


def check_lp_depth(args):
    ok_all, msgs = True, []
    bb = backbone()
    for train in (False, True):
        ds = make_ds('R1', 'train' if train else 'test', train, return_depth=True)
        np.random.seed(4)
        imgs, mats, _, _, _, _, depth = batch_of(ds, 41, True)
        corr = U.frustum_correspondences(bb, mats)
        rgb = _unit(imgs[:, 0])
        dense, real = zip(*[_dense(depth[0, 0, n]) for n in range(depth.shape[2])])
        dense = torch.stack(dense)[None]
        real = torch.stack(real)
        ssim = U.SSIM()
        res = {}
        for name, dep in [('x0.70', dense * 0.7), ('x0.85', dense * 0.85), ('x1.00', dense), ('x1.15', dense * 1.15),
                          ('x1.30', dense * 1.3), ('10 m', torch.full_like(dense, 10.0))]:
            maps = U.photometric_error_maps(dep, rgb, *U.bevdet_cams(mats), corr['pairs'], ssim)
            vals = [pe[v & real[i]] for (pe, v), (i, j) in zip(maps, corr['pairs'][0]) if (v & real[i]).any()]
            res[name] = torch.cat(vals).mean().item()
        best = min(res, key=res.get)
        ok_all &= best == 'x1.00'
        msgs.append(f'{"train (augmented)" if train else "test-time"} {len(corr["pairs"][0])} pairs: ' +
                    ', '.join(f'{k} {v:.4f}' for k, v in res.items()) + f' -> lowest {best}')
    report('lp-depth', ok_all, ' | '.join(msgs))


def check_lp_grad(args):
    ds = make_ds('R1', 'train', True)
    np.random.seed(5)
    imgs, mats = batch_of(ds, 51, False)[:2]
    bb = backbone()
    corr = U.frustum_correspondences(bb, mats)
    N = imgs.shape[2]
    D, fH, fW, _ = bb.frustum.shape
    logits = torch.randn(N, D, fH, fW, requires_grad=True)
    k = bb.depth_scale(mats['intrin_mats'][:, 0], mats['ida_mats'][:, 0])
    loss = U.photometric_loss_paper(logits, imgs[:, 0], U.bevdet_cams(mats), corr['pairs'], k, D_BOUND, U.SSIM())
    loss.backward()
    g = logits.grad
    nz = (g.abs().sum(1) > 0).float().mean().item()
    ok = bool(torch.isfinite(loss)) and bool(torch.isfinite(g).all()) and nz > 0
    report('lp-grad', ok, f'loss {loss.item():.4f}, gradient finite, {100 * nz:.1f}% of feature pixels get a '
           f'gradient, |grad| max {g.abs().max():.2e}')


CHECKS = {'test-inputs': check_test_inputs, 'ext-aug': check_ext_aug, 'corr': check_corr,
          'ld-target': check_ld_target, 'lp-identity': check_lp_identity, 'lp-depth': check_lp_depth,
          'lp-grad': check_lp_grad}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', default=None, choices=sorted(CHECKS))
    args = ap.parse_args()
    torch.manual_seed(0)
    for name in args.only or CHECKS:
        CHECKS[name](args)
    print('ALL PASS' if all(RESULTS.values()) else f'FAILED: {[k for k, v in RESULTS.items() if not v]}')
    return 0 if all(RESULTS.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
