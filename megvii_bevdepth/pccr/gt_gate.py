# PCCR port: new file. GT-as-prediction gate for the evaluation bridge.
"""Runs the test split of a rig through the exp's test dataset (PCCRDetDataset in test mode, the same img_metas the
model's predictions get), takes each sample's GT in the training target format (get_gt: ego frame of the key sample,
[x, y, z, l, w, h, yaw, vx, vy], boxes without LiDAR/radar points dropped as in training) as the predictions with score
1, converts them with the evaluator's _format_bbox (ego -> global, nuScenes json) and scores them with PCCR's devkit.
A correct bridge gives mAP7 = 1.000.
    python pccr/gt_gate.py --info_root data/pccr_megvii --rig R1 --out outputs/gt_gate/plain
"""
import argparse
import os
import sys
import time
from functools import partial

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from bevdepth.datasets.pccr_det_dataset import PCCR_CLASSES, collate_fn  # noqa: E402
from bevdepth.evaluators.pccr_det_evaluator import PCCRDetEvaluator  # noqa: E402
from bevdepth.exps.pccr.base_exp_pccr import build_pccr_dataset, pccr_confs, rig_dir  # noqa: E402
from bevdepth.exps.pccr.base_exp_pccr_canon import build_pccr_canon_dataset, make_canon_conf  # noqa: E402
from bevdepth.datasets.pccr_canon_dataset import canon_collate_fn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--info_root', default='data/pccr_megvii')
    ap.add_argument('--pccr_root', default='/data4/pccr/data')
    ap.add_argument('--rig', default='R1')
    ap.add_argument('--final_h', type=int, default=256)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--canon', default='none', choices=['none', 'plain', 'full'],
                    help="canonical-camera test dataset with this elevation band (base_exp_pccr_canon)")
    args = ap.parse_args()
    _, _, ida, bda = pccr_confs(args.final_h)
    info_path = os.path.join(args.info_root, f'{args.rig}_infos_test.pkl')
    collate = collate_fn
    if args.canon == 'none':
        ds = build_pccr_dataset(info_path, rig_dir(args.pccr_root, args.rig), ida, bda, is_train=False,
                                return_depth=False)
    else:
        canon_conf = make_canon_conf(os.path.join(args.info_root, 'R1_infos_train.pkl'), ida, band=args.canon)
        ds = build_pccr_canon_dataset(info_path, rig_dir(args.pccr_root, args.rig), ida, bda, canon_conf,
                                      is_train=False, return_depth=False)
        collate = canon_collate_fn
    print(f'{args.rig}: {len(ds)} samples, cams {list(ds.ida_aug_conf["cams"])}, H x W {ds.ida_aug_conf["H"]} x '
          f'{ds.ida_aug_conf["W"]}, final_dim {ds.ida_aug_conf["final_dim"]}', flush=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=args.workers,
                                         collate_fn=partial(collate, is_return_depth=False))
    results, metas, idx, n_box, t0 = [], [], 0, 0, time.time()
    for imgs, mats, _, img_metas, _, _ in loader:
        if args.canon == 'none':
            assert tuple(imgs.shape[-2:]) == tuple(ds.ida_aug_conf['final_dim'])
        elif idx == 0:
            print(f'canonical input {tuple(imgs.shape)} (B, sweeps, views, 3, Hp, Wp), {img_metas[0]["canon"]}',
                  flush=True)
        for m in img_metas:
            info = ds.infos[idx]
            assert m['token'] == info['sample_token']
            boxes, labels = ds.get_gt(info, ds.ida_aug_conf['cams'])
            boxes = boxes.numpy().reshape(-1, 9)
            results.append([boxes, np.ones(len(boxes), np.float32), labels.numpy().astype(np.int64)])
            metas.append(m)
            n_box += len(boxes)
            idx += 1
    print(f'{idx} samples through the test pipeline in {time.time() - t0:.0f} s, {n_box} GT boxes as predictions',
          flush=True)
    ev = PCCRDetEvaluator(class_names=list(PCCR_CLASSES), output_dir=os.path.join(args.out, args.rig),
                          rig_dir=rig_dir(args.pccr_root, args.rig))
    ev.evaluate(results, metas)
    m = ev.last_metrics
    print(f'GATE {args.info_root} canon={args.canon} {args.rig} final_h {args.final_h}: mAP7 {m["mAP7"]:.4f} mAP {m["mAP"]:.4f} '
          f'NDS {m["NDS"]:.4f} | ' + ' '.join(f'{c} {v:.3f}' for c, v in m['ap'].items()), flush=True)


if __name__ == '__main__':
    main()
