"""Checks for the Megvii BEVDepth PCCR port (run in the gen2e-megvii env, CPU).

    CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/gen2e-megvii/bin/python -I tools/check_megvii_pccr.py eval-gt RIG [--split test]
    CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/gen2e-megvii/bin/python -I tools/check_megvii_pccr.py render RIG INDEX OUT.jpg [--seed S]

eval-gt  Frame and output chain. The GT that Megvii's NuscDetDataset.get_gt trains on (ego frame,
         zero-point and non-class boxes dropped) is fed as predictions (distinct scores) through
         Megvii's own result formatting (DetNuscEvaluator._format_bbox via the PCCR exp's evaluator,
         with img_metas built as the dataset does) and scored by tools/score_megvii_pccr.py
         (PCCR's evaluation, gen2e-bevdepth env). Expect mAP7 ~100 and zero box errors.
render   Camera geometry as the model sees it: one training sample of the PCCR exp (image
         augmentation, BDA, LiDAR depth labels); GT boxes are projected through inverse BDA,
         inverse sensor2ego, the intrinsics and the ida matrix onto the augmented images.
"""
import argparse
import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MEGVII = os.path.join(REPO, 'megvii_bevdepth')
EXP = os.path.join(MEGVII, 'bevdepth/exps/pccr/bev_depth_lss_r50_256x704_128x128_24e_pccr.py')
BEVDEPTH_PY = os.path.expanduser('~/miniconda3/envs/gen2e-bevdepth/bin/python')


def load_exp():
    sys.path.insert(0, MEGVII)
    spec = importlib.util.spec_from_file_location('pccr_exp', EXP)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def dataset(m, rig, split, is_train):
    from bevdepth.datasets.nusc_det_dataset import NuscDetDataset
    from bevdepth.exps.nuscenes.base_exp import bda_aug_conf, img_conf
    ida = copy.deepcopy(m.pccr_ida_aug_conf)
    ida['cams'] = m.rig_cams(rig, split)
    ida['Ncams'] = len(ida['cams'])
    return NuscDetDataset(ida_aug_conf=ida, bda_aug_conf=bda_aug_conf, classes=m.PCCR_CLASSES,
                          data_root=os.path.join(m.PCCR_ROOT, rig), info_paths=m.info_path(rig, split),
                          is_train=is_train, img_conf=img_conf, return_depth=is_train), ida['cams']


def cmd_eval_gt(args):
    m = load_exp()
    ds, cams = dataset(m, args.rig, args.split, is_train=False)
    results, metas, n = [], [], 0
    for info in ds.infos:
        boxes, labels = ds.get_gt(info, cams)
        boxes, labels = boxes.numpy().reshape(-1, 9), labels.numpy()
        scores = 1.0 - 1e-6 * np.arange(n, n + len(boxes))  # distinct (devkit tie artifact)
        n += len(boxes)
        results.append((boxes, scores, labels))
        metas.append(dict(
            token=info['sample_token'],
            ego2global_translation=np.mean([info['cam_infos'][c]['ego_pose']['translation'] for c in cams], 0),
            ego2global_rotation=np.mean([info['cam_infos'][c]['ego_pose']['rotation'] for c in cams], 0)))
    out = tempfile.mkdtemp(prefix=f'megvii_evalgt_{args.rig}_')
    ev = m.PCCREvaluator(class_names=m.PCCR_CLASSES, output_dir=out)
    path = ev._format_bbox(results, metas, None)
    print(f'{args.rig} {args.split}: {len(results)} samples, {n} GT boxes -> {path}')
    r = subprocess.run([BEVDEPTH_PY, '-I', os.path.join(REPO, 'tools/score_megvii_pccr.py'), path,
                        args.rig, '--split', args.split], capture_output=True, text=True,
                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    print(r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-2000:])
    summ = json.load(open(os.path.join(out, 'metrics_summary.json')))
    # per-class TP errors of the 7 reported classes (the 9-class mean includes traffic_light,
    # which has no GT box with points, so the devkit sets its errors to 1)
    lt = summ['label_tp_errors']
    tp = {k: max(lt[c][k] for c in m.PCCR_CLASSES[:7]) for k in ('trans_err', 'scale_err', 'orient_err')}
    print('7-class TP errors (max over classes):', {k: round(v, 4) for k, v in tp.items()})
    ok = r.returncode == 0 and all(
        sum(summ['label_aps'][c].values()) / 4 > 0.98 for c in m.PCCR_CLASSES[:7]) and tp['trans_err'] < 0.01 \
        and tp['scale_err'] < 1e-3 and tp['orient_err'] < 0.01
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


def cmd_render(args):
    import cv2
    import torch
    m = load_exp()
    ds, cams = dataset(m, args.rig, 'train' if args.rig == 'R1' else 'test', is_train=True)
    np.random.seed(args.seed)
    (imgs, s2e, intrin, ida, _, bda, _, metas, gt_boxes, gt_labels, depth) = ds[args.index]
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    corners = LiDARInstance3DBoxes(gt_boxes[:, :7], origin=(0.5, 0.5, 0.5)).corners  # (N, 8, 3)
    pts = torch.cat([corners.reshape(-1, 3), torch.ones(corners.shape[0] * 8, 1)], 1).T
    ego = torch.inverse(bda) @ pts
    mean = np.array([123.675, 116.28, 103.53]); std = np.array([58.395, 57.12, 57.375])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    tiles = []
    for c in range(imgs.shape[1]):
        img = imgs[0, c].permute(1, 2, 0).numpy() * std + mean  # network tensor (BGR order, see to_rgb)
        img = np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))
        cam = torch.inverse(s2e[0, c]) @ ego
        uv = intrin[0, c] @ cam
        z = uv[2].clone()
        uv = torch.stack([uv[0] / z.clamp(min=1e-6), uv[1] / z.clamp(min=1e-6), torch.ones_like(z), torch.ones_like(z)])
        uv = (ida[0, c] @ uv)[:2].T.reshape(-1, 8, 2).numpy()
        zc = cam[2].reshape(-1, 8).numpy()
        for box, zb in zip(uv, zc):
            if (zb <= 0.1).any():
                continue
            for a, b in edges:
                cv2.line(img, tuple(box[a].astype(int)), tuple(box[b].astype(int)), (0, 255, 0), 1)
        d = depth[0, c].numpy()
        vs, us = np.nonzero(d)
        col = cv2.applyColorMap(np.clip(d[vs, us] / 60 * 255, 0, 255).astype(np.uint8)[:, None], cv2.COLORMAP_JET)
        for u, v, cc in zip(us[::3], vs[::3], col[::3, 0]):
            cv2.circle(img, (int(u), int(v)), 1, tuple(int(x) for x in cc), -1)
        cv2.putText(img, cams[c], (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(img)
    cols = 2
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(args.out, grid)
    print(f'{args.rig} #{args.index}: {len(gt_boxes)} boxes, {imgs.shape[1]} cams, BDA\n{bda.numpy().round(3)}\n-> {args.out}')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('eval-gt')
    e.add_argument('rig')
    e.add_argument('--split', default='test')
    r = sub.add_parser('render')
    r.add_argument('rig')
    r.add_argument('index', type=int)
    r.add_argument('out')
    r.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    return {'eval-gt': cmd_eval_gt, 'render': cmd_render}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
