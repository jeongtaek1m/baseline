# PCCR port: copied unchanged from CoIn3D/pccr/summarize_pccr.py (table of <dir>/<rig>/eval.json).
"""Table of PCCR scores (pccr/eval_pccr_devkit.py's eval.json per rig) for one or more result folders:
  python pccr/summarize_pccr.py BEVDet/logs/pccr_test/cdaA [BEVDet/logs/pccr_test/cdaB ...] [--metric mAP7]
Prints per rig mAP (9 classes), mAP7 (7 dynamic classes) and NDS for each folder, and the mean over the rigs other
than R1 (zero-shot transfer) next to R1 (the training rig), plus per-class AP on R1.
"""
import argparse
import json
import os

RIGS = ['R1', 'R1-t', 'R1-r', 'R1-f', 'R1-c6', 'R1-c10', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    args = ap.parse_args()
    for d in args.dirs:
        res = {}
        for rig in RIGS:
            f = os.path.join(d, rig, 'eval.json')
            if os.path.exists(f):
                with open(f) as fh:
                    res[rig] = json.load(fh)
        print(f'== {d}: {len(res)} of {len(RIGS)} rigs scored')
        print(f'{"rig":8s} {"mAP":>6s} {"mAP7":>6s} {"NDS":>6s}')
        for rig, r in res.items():
            print(f'{rig:8s} {100 * r["mAP"]:6.1f} {100 * r["mAP7"]:6.1f} {100 * r["NDS"]:6.1f}')
        other = [r for rig, r in res.items() if rig != 'R1']
        if other:
            m = {k: 100 * sum(r[k] for r in other) / len(other) for k in ('mAP', 'mAP7', 'NDS')}
            print(f'{"mean!R1":8s} {m["mAP"]:6.1f} {m["mAP7"]:6.1f} {m["NDS"]:6.1f}   ({len(other)} rigs)')
        if 'R1' in res:
            print('R1 AP:', '  '.join(f'{c} {100 * v:.1f}' for c, v in res['R1']['ap'].items()))


if __name__ == '__main__':
    main()
