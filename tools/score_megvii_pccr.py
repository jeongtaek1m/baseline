"""Score a Megvii BEVDepth results_nusc.json on one PCCR rig with PCCR's evaluation.

    ~/miniconda3/envs/gen2e-bevdepth/bin/python -I tools/score_megvii_pccr.py RESULTS_JSON RIG [--split test]

Uses the PCCR adapter we already verified for the BEVDepth rows (bevdepth/mmdet3d/datasets/
pccr_dataset.py, a verbatim copy of PCCR's evaluation; tools/check_bevdepth_pccr.py eval-gt) with
PCCR's 9 classes, because the Megvii run trains PCCR's 9 classes (decision 2026-11-23). Per-class
APs do not depend on the other classes, so the 7-class mAP equals a 7-class evaluation.
metrics_summary.json and metrics_details.json are written next to RESULTS_JSON.
"""
import argparse
import copy
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CLASSES9 = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child',
            'traffic_sign', 'traffic_light']
CLASSES7 = CLASSES9[:7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('rig')
    ap.add_argument('--split', default='test')
    args = ap.parse_args()
    res = os.path.abspath(args.results)
    os.chdir(os.path.join(REPO, 'bevdepth'))
    sys.path.insert(0, os.getcwd())
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    cfg = Config.fromfile('configs/gen2e/bevdepth-r50-pccr.py')
    d = copy.deepcopy(cfg.data.test)
    root = f'../data/pccr/{args.rig}/'
    d.update(data_root=root, ann_file=f'{root}{args.rig}_infos_{args.split}.pkl', pipeline=[],
             classes=CLASSES9)
    ds = build_dataset(d)
    tokens = {i['token'] for i in ds.data_infos}
    pred = json.load(open(res))['results']
    missing, extra = tokens - set(pred), set(pred) - tokens
    if missing or extra:
        raise SystemExit(f'{args.rig}: results cover {len(pred)} samples, missing {len(missing)}, '
                         f'extra {len(extra)} (split {args.split} has {len(tokens)})')
    ds._evaluate_single(res)
    m = json.load(open(os.path.join(os.path.dirname(res), 'metrics_summary.json')))
    ap7 = {c: 100 * sum(m['label_aps'][c].values()) / 4 for c in CLASSES7}
    print(f'{args.rig} {args.split}: mAP7 {sum(ap7.values()) / 7:.3f} | ' +
          ' '.join(f'{c} {v:.1f}' for c, v in ap7.items()))


if __name__ == '__main__':
    main()
