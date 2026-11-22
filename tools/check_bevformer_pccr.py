"""Checks for the BEVFormer PCCR port (run in the gen2e-bevformer env, from anywhere).

    python -I tools/check_bevformer_pccr.py boxes   RIG SPLIT [--samples N]
    python -I tools/check_bevformer_pccr.py eval-gt RIG SPLIT
    python -I tools/check_bevformer_pccr.py render  RIG SPLIT INDEX OUT.jpg

boxes    Box conversion against the nuScenes devkit. For each sample, the boxes that
         PCCRBEVFormerDataset produces (mmdet3d 0.17.1 LiDARInstance3DBoxes) are compared corner
         by corner with the devkit's own boxes from the annotation tables, moved into the
         LIDAR_TOP frame by NuScenes.get_sample_data (then into the config's reference frame
         with the devkit's calibrated_sensor table). Independent of PCCR's info converter.
         mmdet3d boxes and the nuScenes metric only have yaw, so the devkit boxes are compared
         with pitch and roll removed; the number of strongly tilted annotations is printed
         (PCCR has e.g. lying pedestrians).
eval-gt  Output chain: ground-truth boxes are fed as predictions (score 1) through
         _format_bbox and the PCCR evaluation. Expect mAP 1.0, ATE/ASE 0 and AOE ~1e-3 rad (tilted
         annotations). Boxes without LiDAR points are left out, as the devkit drops such GT boxes.
render   Camera geometry: projects the GT boxes with the pipeline's lidar2img onto the images
         produced by the training pipeline (without photometric distortion) and saves a grid.

The dataset is built from projects/configs/gen2e/bevformer_small_pccr.py with data_root and
ann_file replaced by data/pccr/RIG/RIG_infos_SPLIT.pkl.
"""
import argparse
import copy
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BEVFORMER = REPO / 'bevformer'
CONFIG = 'projects/configs/gen2e/bevformer_small_pccr.py'


def build_dataset(rig, split, test_mode, pipeline=None):
    os.chdir(BEVFORMER)  # the config's data_root and checkpoint paths are relative to bevformer/
    sys.path.insert(0, str(BEVFORMER))
    from mmcv import Config
    from mmdet3d.datasets import build_dataset as _build
    cfg = Config.fromfile(CONFIG)  # also runs custom_imports (registers PCCRBEVFormerDataset)
    dcfg = copy.deepcopy(cfg.data.train if not test_mode else cfg.data.test)
    root = f'../data/pccr/{rig}/'
    dcfg.update(data_root=root, ann_file=f'{root}{rig}_infos_{split}.pkl', test_mode=test_mode)
    if test_mode:
        dcfg.pop('samples_per_gpu', None)
    if pipeline is not None:
        dcfg['pipeline'] = pipeline
    return _build(dcfg), cfg


def cmd_boxes(args):
    from nuscenes import NuScenes
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion
    ds, _ = build_dataset(args.rig, args.split, test_mode=False, pipeline=[])
    nusc = NuScenes(version=ds.version, dataroot=ds.data_root, verbose=False)
    idxs = np.linspace(0, len(ds) - 1, min(args.samples, len(ds))).astype(int)
    worst_center, worst_corner, n, tilted = 0.0, 0.0, 0, 0
    for i in idxs:
        info = ds.data_infos[i]
        ours = ds.get_ann_info(i)['gt_bboxes_3d']
        if len(ours) == 0:
            continue
        sample = nusc.get('sample', info['token'])
        lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        _, ref_boxes, _ = nusc.get_sample_data(lidar_sd['token'])
        if ds.ref2ego_translation is not None:
            # LIDAR_TOP -> ego with the devkit's calibrated_sensor table, then -> reference frame
            cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
            for b in ref_boxes:
                b.rotate(Quaternion(cs['rotation']))
                b.translate(np.array(cs['translation']) - ds.ref2ego_translation)
        ref_centers = np.stack([b.center for b in ref_boxes])
        ref_corners = []  # (8, 3) each, yaw only
        for b in ref_boxes:
            yaw, pitch, roll = b.orientation.yaw_pitch_roll
            flat = Box(b.center, b.wlh, Quaternion(axis=[0, 0, 1], radians=yaw))
            ref_corners.append(flat.corners().T)
            tilted += int(max(abs(pitch), abs(roll)) > np.radians(10))
        our_centers = ours.gravity_center.numpy()
        our_corners = ours.corners.numpy()  # (N, 8, 3)
        for c, k in zip(our_centers, our_corners):
            j = int(np.argmin(np.linalg.norm(ref_centers - c, axis=1)))
            worst_center = max(worst_center, float(np.linalg.norm(ref_centers[j] - c)))
            # corner order differs between the libraries: compare as point sets
            d = np.linalg.norm(k[:, None, :] - ref_corners[j][None, :, :], axis=-1)
            worst_corner = max(worst_corner, float(d.min(axis=1).max()), float(d.min(axis=0).max()))
            n += 1
    frame = 'LIDAR_TOP' if ds.ref2ego_translation is None else f'ref2ego {ds.ref2ego_translation}'
    print(f'{args.rig} {args.split} [{frame}]: {n} boxes in {len(idxs)} samples; '
          f'max center error {worst_center:.2e} m, max corner error {worst_corner:.2e} m '
          f'(yaw only; {tilted} devkit boxes tilted by more than 10 deg)')
    ok = worst_center < 1e-3 and worst_corner < 1e-3
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


def cmd_eval_gt(args):
    import torch
    ds, _ = build_dataset(args.rig, args.split, test_mode=False, pipeline=[])
    results = []
    for i in range(len(ds)):
        info = ds.data_infos[i]
        ann = ds.get_ann_info(i)
        # same mask as mmdet3d 0.17.1 NuScenesDataset.get_ann_info
        mask = info['valid_flag'] if ds.use_valid_flag else info['num_lidar_pts'] > 0
        pts = info['num_lidar_pts'][mask]
        keep = (ann['gt_labels_3d'] >= 0) & (pts > 0)
        boxes = ann['gt_bboxes_3d'][torch.from_numpy(keep)]
        results.append(dict(pts_bbox=dict(
            boxes_3d=boxes,
            scores_3d=torch.ones(len(boxes)),
            labels_3d=torch.from_numpy(ann['gt_labels_3d'][keep]).long())))
    with tempfile.TemporaryDirectory() as tmp:
        metrics = ds.evaluate(results, jsonfile_prefix=tmp)
    pre = 'pts_bbox_NuScenes'
    keys = ['mAP', 'mATE', 'mASE', 'mAOE', 'NDS']
    print(f'{args.rig} {args.split}: ' + ', '.join(f'{k} {metrics[f"{pre}/{k}"]:.4f}' for k in keys))
    # AOE tolerance 0.01 rad: the devkit takes yaw from tilted annotation quaternions and the ego
    # pose has small pitch/roll, so yaw-only boxes give ~1e-3 rad; a w/l or yaw-convention error
    # gives ~1.5 rad.
    ok = (metrics[f'{pre}/mAP'] > 0.999 and metrics[f'{pre}/mATE'] < 1e-3
          and metrics[f'{pre}/mASE'] < 1e-3 and metrics[f'{pre}/mAOE'] < 1e-2)
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


def cmd_render(args):
    import cv2
    _, cfg = build_dataset(args.rig, args.split, test_mode=False, pipeline=[])
    pipeline = [p for p in cfg.train_pipeline if p['type'] not in (
        'PhotoMetricDistortionMultiViewImage', 'NormalizeMultiviewImage',
        'DefaultFormatBundle3D', 'CustomCollect3D')]
    ds, _ = build_dataset(args.rig, args.split, test_mode=False, pipeline=pipeline)
    r = ds.pipeline(_prepared(ds, args.index))
    corners = r['gt_bboxes_3d'].corners.numpy()  # (N, 8, 3), LIDAR_TOP frame
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    tiles = []
    for img, l2i in zip(r['img'], r['lidar2img']):
        img = np.ascontiguousarray(img.astype(np.uint8))
        for box in corners:
            p = np.concatenate([box, np.ones((8, 1))], 1) @ np.asarray(l2i).T
            if (p[:, 2] <= 0.1).any():
                continue
            uv = (p[:, :2] / p[:, 2:3]).astype(int)
            for a, b in edges:
                cv2.line(img, tuple(uv[a]), tuple(uv[b]), (0, 255, 0), 2)
        tiles.append(cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)
    print(f'{args.rig} {args.split} #{args.index}: {len(corners)} boxes, '
          f'{len(r["img"])} cameras, image {r["img"][0].shape} -> {out}')
    return 0


def _prepared(ds, index):
    input_dict = ds.get_data_info(index)
    ds.pre_pipeline(input_dict)
    return input_dict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('boxes')
    b.add_argument('rig')
    b.add_argument('split')
    b.add_argument('--samples', type=int, default=200)
    e = sub.add_parser('eval-gt')
    e.add_argument('rig')
    e.add_argument('split')
    r = sub.add_parser('render')
    r.add_argument('rig')
    r.add_argument('split')
    r.add_argument('index', type=int)
    r.add_argument('out')
    args = ap.parse_args()
    return {'boxes': cmd_boxes, 'eval-gt': cmd_eval_gt, 'render': cmd_render}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
