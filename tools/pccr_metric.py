#!/usr/bin/env python3
"""Metric used in the Gen2E main table on PCCR: 7-class mAP.

The four dagger rows of the Gen2E table (BEVDet, BEVFusion-C, Fast-BEV, PETR) are NOT the
official PCCR 9-class mAP. They equal the nuScenes-style mAP averaged over the 7 classes
car, truck, bus, bicycle, motorcycle, adult, child (traffic_sign / traffic_light dropped),
i.e. mean over classes of the mean AP at center-distance thresholds 0.5/1/2/4 m.
Train on R1, evaluate on the *test* split of every rig.

Subcommands
  verify-dagger            recompute the dagger rows from the PCCR repo's released per-class APs
                           (third_party/pccr/models/results/<Model>/trained_on_R1.json)
  summarize FILE [FILE..]  7-class / 9-class mAP and NDS from nuScenes-devkit metrics_summary.json
                           files (one per tested rig; the rig name is taken from the path)

Examples
  python3 tools/pccr_metric.py verify-dagger
  python3 tools/pccr_metric.py summarize work_dirs/bevdepth/R1/test_*_on_*_test/**/metrics_summary.json
"""
import argparse
import glob
import json
import os
import re
import sys

RIGS = ['R1', 'R1-c6', 'R1-c10', 'R1-f', 'R1-r', 'R1-t',
        'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9']
CLASSES9 = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child',
            'traffic_sign', 'traffic_light']
CLASSES7 = CLASSES9[:7]
THRESHOLDS = ['0.5', '1.0', '2.0', '4.0']

# Gen2E main table (trained on R1), dagger rows, columns in RIGS order.
GEN2E_TABLE = {
    'BEVDet':    [25.2, 23.2, 27.1, 0.5, 24.7, 21.7, 1.3, 9.0, 25.5, 0.0, 8.2, 11.2, 4.9, 5.1],
    'BEVFusion': [26.0, 24.0, 27.1, 0.9, 24.0, 21.7, 2.2, 14.6, 25.7, 0.0, 10.6, 12.8, 3.6, 6.5],
    'Fast-BEV':  [12.9, 12.5, 19.2, 6.0, 17.6, 18.3, 7.0, 9.0, 6.1, 1.2, 0.0, 9.0, 1.8, 0.0],
    'PETR':      [21.1, 18.8, 21.0, 2.3, 18.2, 13.4, 1.8, 10.7, 17.0, 0.1, 0.0, 5.8, 1.8, 0.0],
}


def class_ap_from_raw(raw, cls):
    """Mean AP over the 4 distance thresholds from a PCCR results 'raw' dict (two key styles)."""
    vals = []
    for t in THRESHOLDS:
        for key in (f'pts_bbox_NuScenes/{cls}_AP_dist_{t}', f'object/{cls}_ap_dist_{t}'):
            if key in raw:
                vals.append(raw[key] or 0.0)
                break
    if len(vals) != len(THRESHOLDS):
        raise KeyError(f'missing AP entries for class {cls}')
    return sum(vals) / len(vals)


def map_over(aps, classes):
    return 100.0 * sum(aps[c] for c in classes) / len(classes)


def cmd_verify_dagger(args):
    ok = True
    for model, row in GEN2E_TABLE.items():
        path = os.path.join(args.results_dir, model, 'trained_on_R1.json')
        res = json.load(open(path))['results']
        pred = []
        for rig in RIGS:
            raw = res[rig]['metrics']['raw']
            aps = {c: class_ap_from_raw(raw, c) for c in CLASSES7}
            pred.append(map_over(aps, CLASSES7))
        err = max(abs(p - t) for p, t in zip(pred, row))
        ok &= err <= 0.051
        print(f'{model:10s} max|7-class mAP - table| = {err:.3f}')
        print('   7cls :', ' '.join(f'{p:5.1f}' for p in pred))
        print('   table:', ' '.join(f'{t:5.1f}' for t in row))
    print('RESULT:', 'all dagger rows reproduced (<= rounding)' if ok else 'MISMATCH')
    return 0 if ok else 1


def rig_from_path(path):
    m = re.search(r'_on_(R\d(?:-[a-z]\d*)?)_', path) or re.search(r'/(R\d(?:-[a-z]\d*)?)/', path)
    return m.group(1) if m else os.path.dirname(path)


def cmd_summarize(args):
    rows = []
    files = []
    for pattern in args.files:
        files.extend(sorted(glob.glob(pattern, recursive=True)) or [pattern])
    for f in files:
        m = json.load(open(f))
        label_aps = m['label_aps']  # {cls: {'0.5': ap, ...}}
        aps = {c: sum(label_aps[c][t] for t in THRESHOLDS) / len(THRESHOLDS)
               for c in CLASSES9 if c in label_aps}
        missing = [c for c in CLASSES7 if c not in aps]
        if missing:
            print(f'skip {f}: missing classes {missing}', file=sys.stderr)
            continue
        map9 = map_over(aps, [c for c in CLASSES9 if c in aps])
        rows.append((rig_from_path(f), map_over(aps, CLASSES7), map9,
                     100.0 * m.get('nd_score', float('nan')), f, checkpoint_of(f)))
    order = {r: i for i, r in enumerate(RIGS)}
    rows.sort(key=lambda r: order.get(r[0], 99))
    print(f"{'rig':8s} {'mAP7':>6s} {'mAP9':>6s} {'NDS':>6s}  {'checkpoint':16s} file")
    for rig, m7, m9, nds, f, ckpt in rows:
        print(f'{rig:8s} {m7:6.1f} {m9:6.1f} {nds:6.1f}  {os.path.basename(ckpt):16s} {f}')
    if rows:
        # one column per rig in table order; a rig without a result is '--' so columns never shift
        by_rig = {r[0]: r[1] for r in rows}
        print('table row (mAP7):', ' & '.join(f'{by_rig[r]:.1f}' if r in by_rig else '--' for r in RIGS))
        missing = [r for r in RIGS if r not in by_rig]
        if missing:
            print(f'warning: no result for {" ".join(missing)}', file=sys.stderr)
        ckpts = sorted({r[5] for r in rows})
        if len(ckpts) > 1:
            print(f'warning: results come from different checkpoints: {ckpts}', file=sys.stderr)
    return 0


def checkpoint_of(path):
    """Checkpoint recorded by tools/*_test_rigs.sh in the test_on_<RIG>_test directory."""
    d = os.path.dirname(os.path.abspath(path))
    while d != os.path.dirname(d):
        if re.match(r'test_on_.*_test$', os.path.basename(d)):
            f = os.path.join(d, 'checkpoint.txt')
            return open(f).read().strip() if os.path.exists(f) else 'unknown'
        d = os.path.dirname(d)
    return 'unknown'


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('verify-dagger')
    v.add_argument('--results-dir', default=os.path.join(here, '..', 'third_party', 'pccr', 'models', 'results'))
    s = sub.add_parser('summarize')
    s.add_argument('files', nargs='+')
    args = ap.parse_args()
    return cmd_verify_dagger(args) if args.cmd == 'verify-dagger' else cmd_summarize(args)


if __name__ == '__main__':
    sys.exit(main())
