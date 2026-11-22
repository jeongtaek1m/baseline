"""Check the z axis of PCCR ground-truth boxes (frame, center vs bottom, ground contact).

Run it with the gen2e-bevdepth env (numpy, pyquaternion, opencv). The info pkls come from
outside, so they are read with a restricted unpickler that only rebuilds numpy arrays and
plain containers.

    python -I tools/pccr_box_z_check.py stats  [--split test] [--rigs R1 R6 ...]
    python -I tools/pccr_box_z_check.py lidar  [--rigs R1 R4 R6 R9] [--frames 150]
    python -I tools/pccr_box_z_check.py render R1 logs/zcheck/box_z_R1.jpg [--index K]

stats   Box bottom (z - h/2) in the ego frame and box center in the LiDAR frame, per rig and
        class. Also checks that the BEVDet `ann_infos` (ego frame) match `gt_boxes` (LiDAR
        frame) moved by lidar2ego.
lidar   For static, well-observed objects, compares the box bottom with the local ground
        (10th percentile of LiDAR z in a 0.3-1.5 m ring around the footprint) and the box
        top with the highest LiDAR point inside the box.
render  Projects the boxes into every camera of one test frame. Green: z read as the box
        center (correct). Red: z read as the box bottom, i.e. origin=(0.5, 0.5, 0) as in
        PCCR's PETR/BEVFusion loaders.
"""
import argparse
import collections
import json
import os
import pickle

import numpy as np

ROOT = '/data2/PCCR'
RIGS = ['R1', 'R1-c6', 'R1-c10', 'R1-f', 'R1-r', 'R1-t', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9']
CLASSES9 = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child', 'traffic_sign', 'traffic_light']
CLASSES7 = CLASSES9[:7]


# --------------------------------------------------------------------------- safe pkl loading
def _numpy_globals():
    import numpy.core.multiarray as ma
    import numpy.core.numeric as nn
    allowed = {
        ('numpy', 'ndarray'): np.ndarray,
        ('numpy', 'dtype'): np.dtype,
        ('collections', 'OrderedDict'): collections.OrderedDict,
    }
    for mod in ('numpy.core', 'numpy._core'):
        allowed[(f'{mod}.multiarray', '_reconstruct')] = ma._reconstruct
        allowed[(f'{mod}.multiarray', 'scalar')] = ma.scalar
        if hasattr(nn, '_frombuffer'):
            allowed[(f'{mod}.numeric', '_frombuffer')] = nn._frombuffer
    return allowed


class _SafeUnpickler(pickle.Unpickler):
    _allowed = _numpy_globals()

    def find_class(self, module, name):
        obj = self._allowed.get((module, name))
        if obj is None:
            raise pickle.UnpicklingError(f'blocked global {module}.{name}')
        return obj


def load_infos(rig, split):
    with open(f'{ROOT}/{rig}/{rig}_infos_{split}.pkl', 'rb') as f:
        return _SafeUnpickler(f).load()['infos']


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# --------------------------------------------------------------------------- stats
def cmd_stats(args):
    out = {}
    for rig in args.rigs:
        infos = load_infos(rig, args.split)
        l2e_t, rot_dev, ann_diff, ann_len_mismatch = set(), 0.0, 0.0, 0
        acc = {c: collections.defaultdict(list) for c in CLASSES9}
        for info in infos:
            t = np.asarray(info['lidar2ego_translation'], dtype=np.float64)
            R = quat_to_mat(info['lidar2ego_rotation'])
            l2e_t.add(tuple(np.round(t, 3)))
            rot_dev = max(rot_dev, float(np.abs(R - np.eye(3)).max()))
            boxes, names = info['gt_boxes'], info['gt_names']
            if len(boxes) == 0:
                continue
            ctr_ego = boxes[:, :3] @ R.T + t
            for i, name in enumerate(names):
                if name in acc:
                    acc[name]['bottom_ego'].append(ctr_ego[i, 2] - boxes[i, 5] / 2)
                    acc[name]['center_lidar'].append(boxes[i, 2])
                    acc[name]['h'].append(boxes[i, 5])
            if 'ann_infos' in info:
                ann_boxes = np.asarray(info['ann_infos'][0], dtype=np.float64).reshape(-1, 9)
                if len(ann_boxes) == len(boxes):
                    ann_diff = max(ann_diff, float(np.abs(ann_boxes[:, :3] - ctr_ego).max()))
                else:
                    ann_len_mismatch += 1
        res = dict(frames=len(infos), lidar2ego_translation=sorted(l2e_t), lidar2ego_rot_maxdev=rot_dev,
                   ann_infos_vs_gt_boxes_maxdiff=ann_diff, ann_infos_len_mismatch=ann_len_mismatch, classes={})
        print(f'== {rig}/{args.split}: frames={len(infos)} lidar2ego_t={sorted(l2e_t)} rot_dev={rot_dev:.1e} '
              f'ann_infos-gt_boxes max|d|={ann_diff:.1e} len_mismatch={ann_len_mismatch}')
        for c, v in acc.items():
            if not v['h']:
                continue
            be = np.asarray(v['bottom_ego'])
            r = dict(n=len(be), bottom_ego_p5=float(np.percentile(be, 5)), bottom_ego_med=float(np.median(be)),
                     bottom_ego_p95=float(np.percentile(be, 95)), center_lidar_med=float(np.median(v['center_lidar'])),
                     h_med=float(np.median(v['h'])))
            res['classes'][c] = r
            print(f'   {c:13s} n={r["n"]:6d}  bottom(ego) p5/med/p95 = {r["bottom_ego_p5"]:+.3f} {r["bottom_ego_med"]:+.3f} '
                  f'{r["bottom_ego_p95"]:+.3f}   center(lidar) med = {r["center_lidar_med"]:+.3f}   h med = {r["h_med"]:.2f}')
        out[rig] = res
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=1)


# --------------------------------------------------------------------------- lidar
def cmd_lidar(args):
    out = {}
    for rig in args.rigs:
        infos = load_infos(rig, 'test')
        rec = collections.defaultdict(list)
        for k in np.linspace(0, len(infos) - 1, args.frames).astype(int):
            info = infos[k]
            boxes, names = info['gt_boxes'], info['gt_names']
            if len(boxes) == 0:
                continue
            pts = np.fromfile(os.path.join(ROOT, rig, info['lidar_path']), dtype=np.float32).reshape(-1, 5)[:, :3]
            for i, name in enumerate(names):
                x, y, z, l, w, h, yaw = boxes[i]
                speed = np.hypot(*info['gt_velocity'][i])
                if name not in CLASSES7 or np.hypot(x, y) > 25 or info['num_lidar_pts'][i] < 30 or speed > 0.5:
                    continue
                c, s = np.cos(yaw), np.sin(yaw)
                dx, dy = pts[:, 0] - x, pts[:, 1] - y
                lx, ly = c * dx + s * dy, -s * dx + c * dy          # into the box frame
                bottom, top = z - h / 2, z + h / 2
                zwin = (pts[:, 2] > bottom - 1.0) & (pts[:, 2] < top + 0.5)
                foot = (np.abs(lx) <= l / 2) & (np.abs(ly) <= w / 2)
                near = (np.abs(lx) <= l / 2 + 0.3) & (np.abs(ly) <= w / 2 + 0.3)
                ring = (np.abs(lx) <= l / 2 + 1.5) & (np.abs(ly) <= w / 2 + 1.5) & ~near & zwin
                inside = foot & zwin
                if inside.sum() < 10 or ring.sum() < 20:
                    continue
                ground = np.percentile(pts[ring, 2], 10)
                rec[name].append((bottom - ground, top - pts[inside, 2].max()))
        res = {}
        print(f'== {rig}: {args.frames} test frames, static objects within 25 m with >= 30 LiDAR points')
        for name in CLASSES7:
            if not rec[name]:
                continue
            a = np.asarray(rec[name])
            r = dict(n=len(a), bottom_minus_ground_med=float(np.median(a[:, 0])),
                     bottom_minus_ground_p10=float(np.percentile(a[:, 0], 10)),
                     bottom_minus_ground_p90=float(np.percentile(a[:, 0], 90)),
                     top_minus_highest_point_med=float(np.median(a[:, 1])))
            res[name] = r
            print(f'   {name:11s} n={r["n"]:4d}  bottom - local ground: med {r["bottom_minus_ground_med"]:+.3f} '
                  f'[p10 {r["bottom_minus_ground_p10"]:+.3f}, p90 {r["bottom_minus_ground_p90"]:+.3f}]   '
                  f'top - highest point in box: med {r["top_minus_highest_point_med"]:+.3f}')
        out[rig] = res
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=1)


# --------------------------------------------------------------------------- render
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def box_corners(box, z_is_center):
    x, y, z, l, w, h, yaw = box[:7]
    zc = z if z_is_center else z + h / 2
    xs = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * l / 2
    ys = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    zs = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * h / 2
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([x + c * xs - s * ys, y + s * xs + c * ys, zc + zs], 1)


def cmd_render(args):
    import cv2
    infos = load_infos(args.rig, 'test')

    def score(info):
        b, n = info['gt_boxes'], info['gt_names']
        if len(b) == 0:
            return -1
        d = np.hypot(b[:, 0], b[:, 1])
        return 10 * min(np.sum(np.isin(n, ['adult', 'child']) & (d < 15)), 3) + \
            min(np.sum(np.isin(n, ['car', 'truck', 'bus']) & (d < 25)), 6)

    k = args.index if args.index is not None else int(np.argmax([score(i) for i in infos]))
    info = infos[k]
    tiles = []
    for cam, ci in info['cams'].items():
        img = cv2.imread(os.path.join(ROOT, args.rig, ci['data_path']))
        R = np.asarray(ci['sensor2lidar_rotation'])
        t = np.asarray(ci['sensor2lidar_translation'])
        K = np.asarray(ci['cam_intrinsic'])
        for box, name in zip(info['gt_boxes'], info['gt_names']):
            if name not in CLASSES7 or np.hypot(box[0], box[1]) > 40:
                continue
            for z_is_center, color, th in ((False, (0, 0, 255), 1), (True, (0, 255, 0), 2)):
                p = (box_corners(box, z_is_center) - t) @ R           # lidar -> camera
                if np.any(p[:, 2] < 0.5):
                    continue
                uv = p @ K.T
                uv = uv[:, :2] / uv[:, 2:3]
                for a, b in EDGES:
                    cv2.line(img, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)),
                             color, th, cv2.LINE_AA)
        cv2.putText(img, cam, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        tiles.append(cv2.resize(img, (640, 360)))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    mosaic = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)], 0)
    banner = np.zeros((50, mosaic.shape[1], 3), np.uint8)
    cv2.putText(banner, f'{args.rig} test #{k}  green: z read as box center (correct)   '
                        f'red: z read as box bottom (+h/2, PCCR PETR/BEVFusion loaders)',
                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, np.concatenate([banner, mosaic], 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f'{args.out}: {args.rig} test frame {k}, {len(info["cams"])} cameras')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('stats')
    s.add_argument('--split', default='test')
    s.add_argument('--rigs', nargs='+', default=RIGS)
    s.add_argument('--json')
    s.set_defaults(func=cmd_stats)
    s = sub.add_parser('lidar')
    s.add_argument('--rigs', nargs='+', default=['R1', 'R4', 'R6', 'R9'])
    s.add_argument('--frames', type=int, default=150)
    s.add_argument('--json')
    s.set_defaults(func=cmd_lidar)
    s = sub.add_parser('render')
    s.add_argument('rig')
    s.add_argument('out')
    s.add_argument('--index', type=int)
    s.set_defaults(func=cmd_render)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
