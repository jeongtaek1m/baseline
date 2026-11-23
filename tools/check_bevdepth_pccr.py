"""Checks for the BEVDepth PCCR integration (run in the gen2e-bevdepth env, from anywhere).

    python -I tools/check_bevdepth_pccr.py eval-gt RIG SPLIT
    python -I tools/check_bevdepth_pccr.py render  RIG SPLIT INDEX OUT.jpg [--seed S]

eval-gt  Frame and output chain. The ground-truth boxes the training pipeline produces
         (info['ann_infos'] -> LoadAnnotationsBEVDepthUpstream without BDA, i.e. the ego frame
         the model is trained in) are fed as predictions (score 1) through the PCCR adapter's
         _format_bbox (ego -> global) and the PCCR evaluation. A wrong frame, w/l order or yaw
         convention gives ATE in metres or AOE ~1.5 rad. Boxes without LiDAR points are left
         out, as the devkit drops such GT boxes. Expected on R1 test (2026-11-23):
           mAP 0.998: every class 1.000 except bus 0.989 = 1 - 1/90 at all four thresholds;
             the cumulative recall ends a float ulp below 1.0, so the devkit's 101-point
             interpolation sets the last precision to 0. Not a matching error.
           ATE 0.002 m: the adapter (PCCR's code, unchanged) uses box[:3], the bottom centre,
             as the nuScenes centre. On the 1.3% of test samples where the ego vehicle is
             tilted by more than 5 deg (max 11 deg) the h/2 offset along the ego z axis moves
             the centre horizontally, up to 0.41 m for a 4.25 m bus. Re-evaluating the vanilla
             BEVDepth R1 test predictions with gravity centres changes 7-class mAP by +0.02.
           ASE 0, AOE 0.0004 rad.
         Scores are made distinct: with equal scores the devkit's TP-error interpolation over
         confidence returns the error of one box instead of the mean.
render   Camera geometry as the model sees it. Runs the training pipeline of the vanilla
         config (image augmentation, BDA, LiDAR depth labels), projects the BDA-transformed GT
         boxes through inverse BDA, sensor2ego, intrinsics and post_rots/post_trans onto the
         augmented images, overlays the LiDAR depth labels (gt_depth) and saves a grid.

The dataset is built from configs/gen2e/bevdepth-r50-pccr.py with data_root and ann_file
replaced by data/pccr/RIG/RIG_infos_SPLIT.pkl.
"""
import argparse
import copy
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BEVDEPTH = REPO / 'bevdepth'
CONFIG = 'configs/gen2e/bevdepth-r50-pccr.py'


def build_dataset(rig, split, pipeline=None):
    os.chdir(BEVDEPTH)  # the config's data_root is relative to bevdepth/
    sys.path.insert(0, str(BEVDEPTH))
    from mmcv import Config
    from mmdet3d.datasets import build_dataset as _build
    cfg = Config.fromfile(CONFIG)  # also runs custom_imports (registers PCCRDataset etc.)
    dcfg = copy.deepcopy(cfg.data.test)
    root = f'../data/pccr/{rig}/'
    dcfg.update(data_root=root, ann_file=f'{root}{rig}_infos_{split}.pkl')
    if pipeline is not None:
        dcfg['pipeline'] = pipeline
    return _build(dcfg), cfg


def cmd_eval_gt(args):
    import torch
    from mmdet3d.datasets.pipelines import Compose
    ds, cfg = build_dataset(args.rig, args.split, pipeline=[])
    ann = [p for p in cfg.test_pipeline if p['type'] == 'LoadAnnotationsBEVDepthUpstream'][0]
    load_ann = Compose([dict(ann, is_train=False)])
    results = []
    dummy = tuple(torch.zeros(1) for _ in range(6))  # img_inputs placeholders
    # Distinct scores: with all scores equal the devkit's TP-error interpolation over
    # confidence degenerates and reports the error of a single box instead of the mean.
    n_seen = 0
    for i in range(len(ds)):
        info = ds.data_infos[i]
        r = load_ann(dict(ann_infos=copy.deepcopy(info['ann_infos']), img_inputs=dummy))
        boxes, labels = r['gt_bboxes_3d'], torch.as_tensor(r['gt_labels_3d']).long()
        # ann_infos holds the valid_flag boxes in gt_names order (PCCRDataset checks this)
        pts = info['num_lidar_pts'][info['valid_flag']]
        assert len(pts) == len(labels), (len(pts), len(labels))
        keep = (labels >= 0) & torch.from_numpy(pts > 0)
        n = int(keep.sum())
        scores = 1.0 - 1e-6 * torch.arange(n_seen, n_seen + n, dtype=torch.float64)
        n_seen += n
        results.append(dict(pts_bbox=dict(
            boxes_3d=boxes[keep], scores_3d=scores, labels_3d=labels[keep])))
    with tempfile.TemporaryDirectory() as tmp:
        metrics = ds.evaluate(results, jsonfile_prefix=tmp)
    pre = 'pts_bbox_NuScenes'
    keys = ['mAP', 'mATE', 'mASE', 'mAOE', 'NDS']
    print(f'{args.rig} {args.split}: ' + ', '.join(f'{k} {metrics[f"{pre}/{k}"]:.4f}' for k in keys))
    # Tolerances (see the docstring): mAP 0.99 for the devkit's recall-1.0 rounding, ATE 1 cm
    # for the bottom-centre offset on tilted samples, AOE 0.01 rad because ann_infos are
    # yaw-only while the devkit takes yaw from the full annotation quaternion.
    ok = (metrics[f'{pre}/mAP'] > 0.99 and metrics[f'{pre}/mATE'] < 1e-2
          and metrics[f'{pre}/mASE'] < 1e-3 and metrics[f'{pre}/mAOE'] < 1e-2)
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


def cmd_render(args):
    import cv2
    import torch
    _, cfg = build_dataset(args.rig, args.split, pipeline=[])
    keep = ('PrepareImageInputsPCCR', 'LoadAnnotationsBEVDepthUpstream', 'LoadPointsFromFile',
            'PointToMultiViewDepthUpstream')
    pipeline = [p for p in cfg.train_pipeline if p['type'] in keep]
    ds, _ = build_dataset(args.rig, args.split, pipeline=pipeline)
    np.random.seed(args.seed)
    input_dict = ds.get_data_info(args.index)
    ds.pre_pipeline(input_dict)
    r = ds.pipeline(input_dict)
    imgs, rots, trans, intrins, post_rots, post_trans, bda = r['img_inputs']
    corners = r['gt_bboxes_3d'].corners  # (N, 8, 3), ego frame after BDA
    ego = torch.inverse(bda) @ corners.reshape(-1, 3).T  # undo BDA: (3, N*8)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    tiles = []
    for c, canvas in enumerate(r['canvas']):
        img = cv2.cvtColor(np.ascontiguousarray(canvas), cv2.COLOR_RGB2BGR)
        cam = rots[c].T @ (ego - trans[c][:, None])  # sensor2ego inverse
        uvd = intrins[c] @ cam
        uvd = torch.cat([uvd[:2] / uvd[2:3].clamp(min=1e-6), uvd[2:3]], 0)
        uvd = post_rots[c] @ uvd + post_trans[c][:, None]
        uvd = uvd.T.reshape(-1, 8, 3).numpy()
        z = cam.T.reshape(-1, 8, 3)[..., 2].numpy()
        for box, zb in zip(uvd, z):
            if (zb <= 0.1).any():
                continue
            uv = box[:, :2].astype(int)
            for a, b in edges:
                cv2.line(img, tuple(uv[a]), tuple(uv[b]), (0, 255, 0), 1)
        depth = r['gt_depth'][c].numpy()
        vs, us = np.nonzero(depth)
        colors = cv2.applyColorMap(
            np.clip(depth[vs, us] / 60 * 255, 0, 255).astype(np.uint8)[:, None], cv2.COLORMAP_JET)
        for u, v, col in zip(us[::4], vs[::4], colors[::4, 0]):
            cv2.circle(img, (int(u), int(v)), 1, tuple(int(x) for x in col), -1)
        cv2.putText(img, r['cam_names'][c], (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(img)
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)
    print(f'{args.rig} {args.split} #{args.index}: {len(corners)} boxes, {len(r["canvas"])} cameras, '
          f'BDA\n{bda.numpy().round(3)}\n-> {out}')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('eval-gt')
    e.add_argument('rig')
    e.add_argument('split')
    r = sub.add_parser('render')
    r.add_argument('rig')
    r.add_argument('split')
    r.add_argument('index', type=int)
    r.add_argument('out')
    r.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    return {'eval-gt': cmd_eval_gt, 'render': cmd_render}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
