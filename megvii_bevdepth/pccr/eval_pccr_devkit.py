"""PCCR's own evaluation of a nuScenes-format result file: nuscenes-devkit NuScenesEval with the patches of PCCR's
models/BEVDet/pccr_dataset.py (_patched_nuscenes_eval / _evaluate_single / _compute_nds): its 9 classes and class
ranges, the rig's own scenes as the split, PCCR attribute names, NDS without the velocity error. Run it with an
unmodified nuscenes-devkit (the bevf env has 1.1.10), not CoIn3D's extended copy:

  python pccr/eval_pccr_devkit.py <rig_dir> <results_nusc.json> <out_dir> [--version v1.0-test]

Prints and writes <out_dir>/eval.json: mAP (9 classes, PCCR's number), mAP7 (the 7 dynamic classes, the comparison
table's number), NDS, per-class AP (mean over 0.5 / 1 / 2 / 4 m) and per-threshold AP. On identical predictions the
7-class APs equal Gen2E's gen2e/eval_det.py to within 0.001 AP.
"""
import argparse
import json
import os

import numpy as np

CLASSES = ('car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child', 'traffic_sign', 'traffic_light')
CLASS_RANGE = dict(car=50, truck=50, bus=50, bicycle=40, motorcycle=40, adult=40, child=40, traffic_sign=30,
                   traffic_light=30)
ATTR_MAP = {'adult_moving': 'pedestrian.moving', 'adult_standing': 'pedestrian.standing', 'moving': 'vehicle.moving',
            'parked': 'vehicle.parked', 'stopped': 'vehicle.stopped', 'with_rider': 'cycle.with_rider',
            'without_rider': 'cycle.without_rider'}
EVAL_SET = {'v1.0-mini': 'mini_val', 'v1.0-trainval': 'val', 'v1.0-test': 'test'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rig_dir', help='extracted rig folder holding v1.0-*/')
    ap.add_argument('results')
    ap.add_argument('out_dir')
    ap.add_argument('--version', default='v1.0-test')
    ap.add_argument('--scene-prefix', default='',
                    help="evaluate only scenes named <prefix>_* (e.g. 'val' in v1.0-trainval); default: every scene")
    ap.add_argument('--pred-samples-only', action='store_true',
                    help='score only the samples the result file holds (smoke tests on a cut pkl); never for the tables')
    args = ap.parse_args()

    from nuscenes import NuScenes
    from nuscenes.eval.common import loaders as L
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.detection import constants as C, data_classes as DC, evaluate as E, utils as U
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.data_classes import DetectionConfig

    # PCCRDataset.__init__: the config is built before the names are patched, then given PCCR's classes / ranges
    cfg = DetectionConfig.deserialize(config_factory('detection_cvpr_2019').serialize())
    cfg.class_names = list(CLASSES)
    cfg.class_range = dict(CLASS_RANGE)

    nusc = NuScenes(version=args.version, dataroot=args.rig_dir, verbose=False)
    # the evaluated samples: PCCR takes the samples of its pkl, i.e. every scene of the split (trainval holds both
    # train_* and val_* scenes, hence --scene-prefix)
    keep = {s['token'] for s in nusc.scene if not args.scene_prefix or s['name'].split('_')[0] == args.scene_prefix}
    scenes = [s['name'] for s in nusc.scene if s['token'] in keep]
    tokens = [s['token'] for s in nusc.sample if s['scene_token'] in keep]
    if args.pred_samples_only:  # smoke tests on a cut pkl: missing samples would cap the recall below AP's 0.1 floor
        with open(args.results) as f:
            have = set(json.load(f)['results'])
        tokens = [t for t in tokens if t in have]
        scenes = [s['name'] for s in nusc.scene if s['token'] in {nusc.get('sample', t)['scene_token'] for t in tokens}]
        print(f'scoring only the {len(tokens)} samples of the result file ({len(scenes)} scenes)')

    def subset(boxes):  # _patched_nuscenes_eval.subset_eval_boxes(include_empty=True)
        out = EvalBoxes()
        for t in tokens:
            out.add_boxes(t, boxes[t] if t in boxes.sample_tokens else [])
        return out

    load_pred, load_gt = L.load_prediction, L.load_gt
    E.load_prediction = L.load_prediction = lambda *a, **k: (lambda r: (subset(r[0]), r[1]))(load_pred(*a, **k))
    E.load_gt = L.load_gt = lambda *a, **k: subset(load_gt(*a, **k))
    L.create_splits_scenes = lambda: dict(train=scenes, val=scenes, test=scenes, mini_train=scenes, mini_val=scenes)
    mapper = lambda name: name if name in CLASSES else None
    L.category_to_detection_name = U.category_to_detection_name = mapper
    C.DETECTION_NAMES = DC.DETECTION_NAMES = list(CLASSES)
    C.ATTRIBUTE_NAMES = DC.ATTRIBUTE_NAMES = sorted(set(C.ATTRIBUTE_NAMES) | set(ATTR_MAP) | set(ATTR_MAP.values()))
    for attr in nusc.attribute:
        attr['name'] = ATTR_MAP.get(attr['name'], attr['name'])

    os.makedirs(args.out_dir, exist_ok=True)
    E.NuScenesEval(nusc, config=cfg, result_path=args.results, eval_set=EVAL_SET[args.version],
                   output_dir=args.out_dir, verbose=False).main(render_curves=False)
    m = json.load(open(os.path.join(args.out_dir, 'metrics_summary.json')))

    tp = [max(0.0, 1.0 - v) for k, v in m['tp_errors'].items()  # PCCRDataset._compute_nds, velocity discarded
          if k != 'vel_err' and v is not None and np.isfinite(v)]
    nds = (cfg.mean_ap_weight * m['mean_ap'] + sum(tp)) / (cfg.mean_ap_weight + len(tp))
    ap = {c: m['mean_dist_aps'][c] for c in CLASSES}
    res = dict(results=os.path.abspath(args.results), rig_dir=args.rig_dir, version=args.version,
               mAP=m['mean_ap'], mAP7=float(np.mean([ap[c] for c in CLASSES[:7]])), NDS=nds, ap=ap,
               ap_th={c: m['label_aps'][c] for c in CLASSES})
    json.dump(res, open(os.path.join(args.out_dir, 'eval.json'), 'w'), indent=2)
    print(f"{os.path.basename(args.rig_dir.rstrip('/'))} {args.version}: mAP {res['mAP']:.4f} (9 cls)  "
          f"mAP7 {res['mAP7']:.4f}  NDS {nds:.4f} | " + ' '.join(f'{c} {ap[c]:.3f}' for c in CLASSES))


if __name__ == '__main__':
    main()
