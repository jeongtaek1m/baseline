# PCCR port: new file. Visual check of the data pipeline: GT boxes and the online LiDAR depth GT on the network input.
"""For a few samples, takes the dataset output (augmented image, sensor2ego, intrinsic, ida, bda, GT boxes, depth
labels) and projects the GT boxes back with the inverse of the model's own lifting (BaseLSSFPN.get_geometry:
ego_bda = bda @ sensor2ego @ K^-1 @ ida^-1 @ (u d, v d, d)), so boxes land on their objects only if every matrix
the model gets is right. Depth labels are drawn as coloured dots (near red -> far blue). Train mode (R1: random
ida / bda) and test mode (other rigs, GT from get_gt) are both supported.
    python pccr/vis_check.py --info_root data/pccr_megvii --rig R1 --split train --idx 0 1500 --out outputs/vis
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from bevdepth.datasets.pccr_det_dataset import PCCR_CLASSES  # noqa: E402
from bevdepth.exps.pccr.base_exp_pccr import build_pccr_dataset, pccr_confs, rig_dir  # noqa: E402

COLORS = [(0, 255, 0), (0, 128, 255), (255, 0, 255), (255, 255, 0), (0, 255, 255), (0, 0, 255), (255, 128, 0),
          (255, 255, 255), (128, 0, 255)]  # BGR per class
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def corners(b):
    x, y, z, l, w, h, yaw = b[:7]
    c = np.array([[sx * l / 2, sy * w / 2, sz * h / 2] for sz in (-1, 1) for sx, sy in ((1, 1), (1, -1), (-1, -1),
                                                                                       (-1, 1))])
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return c @ R.T + np.array([x, y, z])


def project(pts_ego_bda, bda, s2e, K, ida):
    p = np.concatenate([pts_ego_bda, np.ones((len(pts_ego_bda), 1))], 1).T
    p = np.linalg.inv(s2e) @ np.linalg.inv(bda) @ p  # camera frame
    z = p[2]
    uv = (K[:3, :3] @ p[:3])[:2] / np.maximum(z, 1e-6)
    q = ida @ np.stack([uv[0], uv[1], np.ones_like(z), np.ones_like(z)])
    return q[:2].T, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--info_root', default='data/pccr_megvii')
    ap.add_argument('--pccr_root', default='/data4/pccr/data')
    ap.add_argument('--rig', default='R1')
    ap.add_argument('--split', default='train', choices=['train', 'test'])
    ap.add_argument('--idx', type=int, nargs='+', default=[0])
    ap.add_argument('--final_h', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='outputs/vis')
    args = ap.parse_args()
    np.random.seed(args.seed)
    _, _, ida_conf, bda_conf = pccr_confs(args.final_h)
    info_path = os.path.join(args.info_root, f'{args.rig}_infos_{args.split}.pkl')
    ds = build_pccr_dataset(info_path, rig_dir(args.pccr_root, args.rig), ida_conf, bda_conf,
                            is_train=args.split == 'train', return_depth=True)
    mean = np.array([123.675, 116.28, 103.53], np.float32)
    std = np.array([58.395, 57.12, 57.375], np.float32)
    os.makedirs(args.out, exist_ok=True)
    tag = os.path.basename(os.path.normpath(args.info_root))
    for idx in args.idx:
        (imgs, s2e, intrin, ida, _, bda, _, metas, gt_boxes, gt_labels, depth) = ds[idx]
        cams = list(ds.ida_aug_conf['cams'])
        if args.split == 'test':  # test mode returns no GT: take the training-format GT of the sample
            gt_boxes, gt_labels = ds.get_gt(ds.infos[idx], cams)
        gt_boxes = gt_boxes.numpy().reshape(-1, 9)
        gt_labels = gt_labels.numpy()
        tiles, n_px, n_cells, n_drawn = [], 0, 0, 0
        for n, cam in enumerate(cams):
            img = imgs[0, n].permute(1, 2, 0).numpy() * std + mean  # channel order as fed (to_rgb of a PIL RGB image)
            img = np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))
            d = depth[0, n].numpy()
            H, W = d.shape
            n_px += int((d > 0).sum())
            cells = d.reshape(H // 16, 16, W // 16, 16).max((1, 3))
            n_cells += int((cells > 0).sum())
            ys, xs = np.nonzero(d)
            for y, x in zip(ys, xs):
                t = float(np.clip((d[y, x] - 2) / 56, 0, 1))
                cv2.circle(img, (int(x), int(y)), 1, (int(255 * t), 0, int(255 * (1 - t))), -1)
            for b, lab in zip(gt_boxes, gt_labels):
                uv, z = project(corners(b), bda.numpy(), s2e[0, n].numpy(), intrin[0, n].numpy(), ida[0, n].numpy())
                if (z < 0.5).any() or not ((uv[:, 0] > -W) & (uv[:, 0] < 2 * W) & (uv[:, 1] > -H)
                                           & (uv[:, 1] < 2 * H)).all():
                    continue
                if not ((uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)).any():
                    continue
                n_drawn += 1
                for i, j in EDGES:
                    cv2.line(img, tuple(np.int32(uv[i])), tuple(np.int32(uv[j])), COLORS[int(lab)], 1, cv2.LINE_AA)
            cv2.putText(img, cam, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
        cols = 2
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.concatenate([np.concatenate(tiles[r * cols:(r + 1) * cols], 1) for r in range(len(tiles) // cols)], 0)
        f = os.path.join(args.out, f'{tag}_{args.rig}_{args.split}_{idx}_h{args.final_h}.jpg')
        cv2.imwrite(f, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
        names = ', '.join(f'{PCCR_CLASSES[c]} {int((gt_labels == c).sum())}' for c in np.unique(gt_labels))
        print(f'{f}: {len(cams)} cams {tuple(imgs.shape[-2:])}, bda\n{bda.numpy().round(3)}\n  {len(gt_boxes)} GT boxes '
              f'({names}), {n_drawn} box-views drawn, depth pixels {n_px} ({n_px / (len(cams) * H * W):.2%}), '
              f'16x16 cells with depth {n_cells / (len(cams) * (H // 16) * (W // 16)):.1%}', flush=True)


if __name__ == '__main__':
    main()
