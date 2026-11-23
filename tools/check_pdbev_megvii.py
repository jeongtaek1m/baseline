"""CPU checks for the PD-BEV port on the co-author's Megvii PCCR base (gen2e-megvii env).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 ~/miniconda3/envs/gen2e-megvii/bin/python -I tools/check_pdbev_megvii.py \
        [--range A] [--n 8] [--only NAME ...]

Each check measures the property directly against an independent computation and prints PASS / FAIL:
  test-inputs   the PD-BEV test dataset returns exactly the co-author's PCCRDetDataset images and matrices
  intrinsics    actual intrinsics (PDBEV-9): ida(principal point) == (cx', cy'), |fx'| == resize * fx, incl. flips
  virtual       perturbation within the ranges; PDBEV-10 axes: pitch_aug turns about camera x, roll_aug about camera z
  targets       2D heatmap peaks == independently projected valid objects (PDBEV-4/5/6/7), per view, camera, class;
                every peak has its window in the mask (real 3x3, virtual 2x3); channel 2 == channel 6 * k (virtual depth)
  geometry      the target centre (pixel, virtual depth) through the model's get_geometry with the same camera (real or
                virtual, PDBEV-3) and bda lands on the bda-augmented box centre
  depth-labels  L_pg labels: one fg cell per 2x2 block holding a centre depth in range, bin = floor(min depth) - 1
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
from pyquaternion import Quaternion  # noqa: E402

from bevdepth.datasets.pccr_det_dataset import PCCR_CLASSES  # noqa: E402
from bevdepth.datasets.pccr_pdbev_dataset import (PDBEV_AUG, PCCRPDBEVDataset, collate_fn_pdbev,  # noqa: E402
                                                  virtual_depth_scale)
from bevdepth.exps.nuscenes.base_exp import img_conf  # noqa: E402
from bevdepth.exps.pccr.base_exp_pccr import build_pccr_dataset, ida_conf_for, pccr_confs, rig_dir  # noqa: E402

PCCR_ROOT = os.path.join(REPO, 'data/pccr')
RESULTS = {}


def report(name, ok, msg):
    RESULTS[name] = ok
    print(f'[{"PASS" if ok else "FAIL"}] {name}: {msg}', flush=True)


def make_ds(rig, split, is_train, aug=None, ida_update=None):
    _, _, ida, bda = pccr_confs(256)
    info = f'data/pccr_megvii/{rig}_infos_{split}.pkl'
    conf = ida_conf_for(ida, mmcv.load(info))
    if ida_update:
        conf.update(ida_update)
    return PCCRPDBEVDataset(ida_aug_conf=conf, bda_aug_conf=bda, classes=PCCR_CLASSES,
                            data_root=rig_dir(PCCR_ROOT, rig), info_paths=info, is_train=is_train,
                            return_depth=False, img_conf=img_conf, pdbev_aug=aug)


def check_test_inputs(args):
    bad = []
    for rig in ('R1', 'R6'):
        _, _, ida, bda = pccr_confs(256)
        info = f'data/pccr_megvii/{rig}_infos_test.pkl'
        ref = build_pccr_dataset(info, rig_dir(PCCR_ROOT, rig), ida, bda, is_train=False, return_depth=False,
                                 img_conf=img_conf)
        ds = make_ds(rig, 'test', False)
        for i in (0, 777):
            a, b = ref[i], ds[i]
            for j in (0, 1, 2, 3, 4, 5):
                if not torch.equal(a[j], b[j]):
                    bad.append(f'{rig}#{i} item {j}')
            if a[7]['token'] != b[7]['token']:
                bad.append(f'{rig}#{i} token')
            if set(b[10]) != {'actual_intrin_mats'}:
                bad.append(f'{rig}#{i} extra keys {sorted(b[10])}')
    report('test-inputs', not bad, 'identical images / matrices / tokens on R1, R6' if not bad else str(bad))


def check_intrinsics(args):
    ds = make_ds('R1', 'train', True, PDBEV_AUG[args.range], ida_update=dict(rot_lim=(0.0, 0.0)))
    err_pp, err_f, n_flip, n = 0.0, 0.0, 0, 0
    for i in range(args.n):
        ret = ds[i * 97]
        intr, ida, act = ret[2][0], ret[3][0], ret[10]['actual_intrin_mats'][0]
        for k in range(intr.shape[0]):
            resize, _, _, flip, _ = ds._ida_log[k]
            pp = ida[k, :2, :2] @ intr[k, :2, 2] + ida[k, :2, 3]  # where the optical axis lands in the network image
            err_pp = max(err_pp, float((pp - act[k, :2, 2]).abs().max()))
            err_f = max(err_f, abs(float(act[k, 0, 0]) - resize * float(intr[k, 0, 0])),
                        abs(float(act[k, 1, 1]) - resize * float(intr[k, 1, 1])),
                        abs(abs(float(ida[k, 0, 0])) - resize))
            n_flip += int(flip)
            n += 1
    report('intrinsics', err_pp < 1e-3 and err_f < 1e-3 and n_flip > 0,
           f'{n} cameras ({n_flip} flipped): principal point max err {err_pp:.2e} px, focal max err {err_f:.2e}')


def _euler(R):
    from bevdepth.datasets.pccr_pdbev_dataset import quaternion_to_euler
    return quaternion_to_euler(Quaternion(matrix=R.astype(np.float64), atol=1e-5).elements)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def check_virtual(args):
    aug = PDBEV_AUG[args.range]
    ds = make_ds('R1', 'train', True, aug)
    over = []
    for i in range(args.n):
        ret = ds[i * 131]
        real, virt = ret[1][0].numpy().astype(np.float64), ret[10]['virtual_sensor2ego_mats'][0].numpy()
        for k in range(real.shape[0]):
            d = _wrap(_euler(virt[k, :3, :3]) - _euler(real[k, :3, :3]))  # [pitch_e, yaw, roll_e]
            dt = virt[k, :3, 3] - real[k, :3, 3]
            lim = [aug['roll_aug'], aug['yaw_aug'], aug['pitch_aug']]  # PDBEV-10: pitch_e <- roll_aug, roll_e <- pitch_aug
            for v, (lo, hi), nm in zip(d, lim, ('pitch_e', 'yaw', 'roll_e')):
                if not (lo - 1e-4 <= v <= hi + 1e-4):
                    over.append(f'#{i} cam{k} {nm} {v:.4f}')
            for v, key in zip(dt, ('extri_x_aug', 'extri_y_aug', 'extri_z_aug')):
                lo, hi = aug[key]
                if not (lo - 1e-4 <= v <= hi + 1e-4):
                    over.append(f'#{i} cam{k} {key} {v:.3f}')
    # PDBEV-10 axes: one perturbation at a time, rotation axis of R_real^T R_virtual in camera coordinates
    axes = {}
    zero = dict(pitch_aug=(0.0, 0.0), yaw_aug=(0.0, 0.0), roll_aug=(0.0, 0.0),
                extri_x_aug=(0.0, 0.0), extri_y_aug=(0.0, 0.0), extri_z_aug=(0.0, 0.0))
    for key, want in (('pitch_aug', 0), ('roll_aug', 2), ('yaw_aug', None)):
        a = dict(zero)
        a[key] = (0.1, 0.1)
        ds.pdbev_aug = a
        ret = ds[0]
        real, virt = ret[1][0].numpy().astype(np.float64), ret[10]['virtual_sensor2ego_mats'][0].numpy()
        dom = []
        for k in range(real.shape[0]):
            q = Quaternion(matrix=real[k, :3, :3].T @ virt[k, :3, :3].astype(np.float64), atol=1e-5)
            ax = np.abs(q.axis)
            if want is None:  # yaw: about the ego z axis = camera -y for a level camera
                ax = np.abs(real[k, :3, :3].T @ np.array([0, 0, 1.0]))
                ok_k = np.allclose(np.abs(q.axis), ax, atol=0.02)
            else:
                ok_k = ax[want] > 0.95
            dom.append(ok_k)
        axes[key] = all(dom)
    ok = not over and all(axes.values())
    report('virtual', ok, f'{args.n} samples x 8 cams within range {args.range}' + (f'; out of range {over[:5]}' if over
           else '') + f'; axes {axes}')


def _reference_peaks(boxes, labels, cam2ego, K, ida, H8, W8):
    """Independent projection of the box centres (float64): {(class, v_int, u_int)} of the valid ones, plus the
    positions within 1e-3 px / 1e-3 m of a pixel, map-edge or depth boundary (float32 rounding may go either way)."""
    out, amb, info = set(), [], {}
    if len(boxes) == 0:
        return out, amb, info
    c = boxes[:, :3].numpy().astype(np.float64)  # gravity centre (PCCRDetDataset boxes are centred)
    pc = (np.linalg.inv(cam2ego.astype(np.float64)) @ np.c_[c, np.ones(len(c))].T)[:3].T
    uvz = (K.astype(np.float64) @ pc.T).T
    uv = uvz[:, :2] / uvz[:, 2:3]
    uv = (ida[:2, :2].numpy().astype(np.float64) @ uv.T).T + ida[:2, 3].numpy()
    uv8 = uv / 8
    for j in range(len(c)):
        frac = uv8[j] - np.round(uv8[j])
        if abs(pc[j, 2] - 1.0) < 1e-3 or (np.abs(frac) < 1e-3).any():
            amb.append((int(labels[j]), uv8[j, 1], uv8[j, 0]))
            continue
        if pc[j, 2] > 1.0 and 0 <= uv8[j, 0] < W8 and 0 <= uv8[j, 1] < H8:
            key = (int(labels[j]), int(uv8[j, 1]), int(uv8[j, 0]))
            out.add(key)
            info.setdefault(key, []).append((j, uv[j], pc[j, 2]))
    return out, amb, info


def check_targets_and_geometry(args, do_targets=True, do_geometry=True, do_depth=True):
    aug = PDBEV_AUG[args.range]
    ds = make_ds('R1', 'train', True, aug)
    fH, fW = ds.ida_aug_conf['final_dim']
    H8, W8 = fH // 8, fW // 8
    model_bb = None
    if do_geometry:
        from bevdepth.layers.backbones.pdbev_lss_fpn import PDBEVLSSFPN
        bconf, _, _, _ = pccr_confs(256)
        bconf['d_bound'] = [1.0, 100.0, 1.0]
        bconf['img_backbone_conf'] = dict(bconf['img_backbone_conf'], init_cfg=None)
        model_bb = PDBEVLSSFPN(**bconf).eval()
    miss, extra, win_bad, ch_bad, n_peaks, geo_err, n_geo, dl_bad, n_fg, n_amb = 0, 0, 0, 0, 0, 0.0, 0, 0, 0, 0
    for i in range(args.n):
        ret = ds[i * 113 + 5]
        extra_d = ret[10]
        tg = extra_d['targets_2d']
        bda = ret[5]
        boxes, labels = ds.get_gt(ds.infos[i * 113 + 5], list(ds.ida_aug_conf['cams']))
        boxes, labels = ds.filter_objects(boxes, labels, bda)
        intr, ida, act = ret[2][0], ret[3][0], extra_d['actual_intrin_mats'][0]
        views = (('', ret[1][0], 2), ('_aug', extra_d['virtual_sensor2ego_mats'][0], 1))
        for suffix, s2e, rows_below in views:
            hm, ann, mask = tg['heatmaps_2d' + suffix], tg['ann_maps_2d' + suffix], tg['heatmap_masks_2d' + suffix]
            for k in range(s2e.shape[0]):
                ref, amb, info = _reference_peaks(boxes, labels, s2e[k].numpy(), intr[k, :3, :3].numpy(), ida[k],
                                                  H8, W8)
                got = {(int(c), int(v), int(u)) for v, u, c in zip(*np.nonzero(hm[k].numpy() == 1.0))}
                for (ca, va, ua) in amb:  # boundary cases: drop peaks within a pixel of them from both sides
                    near = {g for g in got | ref if g[0] == ca and abs(g[1] - va) <= 1.5 and abs(g[2] - ua) <= 1.5}
                    got -= near
                    ref -= near
                    n_amb += 1
                if do_targets:
                    miss += len(ref - got)
                    extra += len(got - ref)
                    n_peaks += len(got)
                    for (c, v, u) in ref:
                        win = mask[k, max(v - 1, 0):v + rows_below, max(u - 1, 0):u + 2, c]
                        if not bool((win == 1).all()):
                            win_bad += 1
                    m = mask[k].numpy() == 1  # (H8, W8, C)
                    a = ann[k].numpy()
                    scale = virtual_depth_scale(float(act[k, 0, 0]), float(act[k, 1, 1]))
                    if np.abs(a[:, :, 2, :][m] - a[:, :, 6, :][m] * scale).max(initial=0) > 1e-3:
                        ch_bad += 1
                if do_geometry:
                    pts, want = [], []
                    for (c, v, u), lst in info.items():
                        if len(lst) != 1:
                            continue
                        j, uv, z = lst[0]
                        if abs(float(ann[k, v, u, 6, c]) - z) > 1e-3:  # window overwritten by another object
                            continue
                        pts.append([uv[0], uv[1], float(ann[k, v, u, 2, c]), 1.0])
                        want.append((bda[:3, :3].double() @ boxes[j, :3].double()).numpy())
                    if pts:
                        model_bb.frustum = torch.tensor(pts, dtype=torch.float32).view(len(pts), 1, 1, 4)
                        g = model_bb.get_geometry(s2e[None], intr[None], ida[None], bda[None], act[None])
                        got_xyz = g[0, k, :, 0, 0].double().numpy()
                        geo_err = max(geo_err, float(np.abs(got_xyz - np.stack(want)).max()))
                        n_geo += len(pts)
        if do_depth:
            from bevdepth.models.pdbev_bev_depth import PDBEVBEVDepth

            class _S:
                downsample_from_ann, d_bound, depth_channels = 2, [1.0, 100.0, 1.0], 99
            lab = PDBEVBEVDepth.get_downsampled_gt_depth(_S, tg['ann_maps_2d'][None, :, :, :, 2, :].max(-1).values)
            d = tg['ann_maps_2d'][:, :, :, 2, :].max(-1).values  # (N, H8, W8)
            blocks = d.view(d.shape[0], H8 // 2, 2, W8 // 2, 2).permute(0, 1, 3, 2, 4).reshape(-1, 4)
            bmin = torch.where(blocks > 0, blocks, torch.full_like(blocks, 1e5)).min(-1).values
            want_fg = (bmin >= 1.0) & (bmin < 100.0)
            fg = lab.max(1).values > 0
            n_fg += int(fg.sum())
            if not torch.equal(fg, want_fg):
                dl_bad += 1
            elif fg.any() and not torch.equal(lab[fg].argmax(1), (bmin[fg].floor() - 1).long()):
                dl_bad += 1
    if do_targets:
        report('targets', miss == 0 and extra == 0 and win_bad == 0 and ch_bad == 0 and n_peaks > 0,
               f'{args.n} samples, 2 views x 8 cams: {n_peaks} peaks, missing {miss}, extra {extra}, '
               f'incomplete windows {win_bad}, ch2 != ch6*k in {ch_bad} maps, boundary cases skipped {n_amb}')
    if do_geometry:
        report('geometry', n_geo > 0 and geo_err < 2e-3,
               f'{n_geo} target centres (real + virtual views): max |get_geometry - bda(box centre)| {geo_err:.2e} m')
    if do_depth:
        report('depth-labels', dl_bad == 0 and n_fg > 0, f'{args.n} samples: {n_fg} fg cells, bad samples {dl_bad}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--range', default='A', choices=sorted(PDBEV_AUG))
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--only', nargs='+', default=None)
    args = ap.parse_args()
    np.random.seed(0)
    torch.manual_seed(0)
    want = set(args.only or ['test-inputs', 'intrinsics', 'virtual', 'targets', 'geometry', 'depth-labels'])
    if 'test-inputs' in want:
        check_test_inputs(args)
    if 'intrinsics' in want:
        check_intrinsics(args)
    if 'virtual' in want:
        check_virtual(args)
    if want & {'targets', 'geometry', 'depth-labels'}:
        check_targets_and_geometry(args, 'targets' in want, 'geometry' in want, 'depth-labels' in want)
    # the collate of a training batch (shapes the model reads)
    ds = make_ds('R1', 'train', True, PDBEV_AUG[args.range])
    batch = collate_fn_pdbev([ds[0], ds[1]])
    shapes = {k: tuple(v.shape) for k, v in batch[1].items()}
    shapes.update({k: tuple(v.shape) for k, v in batch[6].items()})
    print('batch shapes:', shapes)
    print('ALL PASS' if all(RESULTS.values()) else f'FAILED: {[k for k, v in RESULTS.items() if not v]}')
    return 0 if all(RESULTS.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
