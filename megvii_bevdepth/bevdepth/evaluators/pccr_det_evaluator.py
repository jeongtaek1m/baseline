# PCCR port: new file. DetNuscEvaluator whose scoring is PCCR's (pccr/eval_pccr_devkit.py).
"""The result file is written by the official _format_bbox (ego-frame boxes -> global frame with the key sample's
ego pose, nuScenes json), with one change: PCCR's classes have no entry in the nuScenes default-attribute table, so
the attribute falls back to '' exactly as the PCCR port of BEVDet does (CoIn3D BEVDet
mmdet3d/datasets/nuscenes_dataset.py, `self.DefaultAttribute.get(name, '')`). Attributes and velocities do not enter
mAP / mAP7. The scoring is pccr/eval_pccr_devkit.py (md5 c8f540b7..., an unmodified nuscenes-devkit 1.1.10 with
PCCR's 9 classes, class ranges and attribute names), run in its own python (PCCR_DEVKIT_PY), on the rig's own
v1.0-test database: <output_dir>/results_nusc.json -> <output_dir>/eval.json.
"""
import json
import os
import os.path as osp
import subprocess

import mmcv
import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from bevdepth.evaluators.det_evaluators import DetNuscEvaluator

__all__ = ['PCCRDetEvaluator']

HERE = osp.dirname(osp.abspath(__file__))
DEVKIT_SCRIPT = osp.join(HERE, '..', '..', 'pccr', 'eval_pccr_devkit.py')
# Poseidon: the python every PCCR row is scored with (nuscenes-devkit 1.1.10); shannon: the bevf env
DEFAULT_DEVKIT_PY = '/data/giyong/envs/bevformer_canon/bin/python'


class PCCRDetEvaluator(DetNuscEvaluator):

    def __init__(self, class_names, output_dir=None, rig_dir=None, version='v1.0-test', devkit_py=None, **kwargs):
        super().__init__(class_names=class_names, output_dir=output_dir, version=version, **kwargs)
        self.rig_dir = rig_dir
        self.devkit_py = devkit_py or os.environ.get('PCCR_DEVKIT_PY', DEFAULT_DEVKIT_PY)
        self.last_metrics = None

    def _evaluate_single(self, result_path, logger=None, metric='bbox', result_name='pts_bbox'):
        out_dir = osp.dirname(osp.abspath(result_path))
        with open(result_path) as f:  # PCCR port: nuscenes-devkit raises on a result file without a single box
            n_boxes = sum(len(v) for v in json.load(f)['results'].values())
        if n_boxes == 0:  # (every score under the head's threshold, e.g. BEVDepth on R5): every AP is 0, NDS 0
            classes = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child', 'traffic_sign', 'traffic_light']
            self.last_metrics = dict(results=osp.abspath(result_path), rig_dir=self.rig_dir, version=self.version,
                                     mAP=0.0, mAP7=0.0, NDS=0.0, ap={c: 0.0 for c in classes},
                                     ap_th={c: {t: 0.0 for t in ('0.5', '1.0', '2.0', '4.0')} for c in classes},
                                     note='no predicted box in the whole result file; devkit not run, all APs 0')
            with open(osp.join(out_dir, 'eval.json'), 'w') as f:
                json.dump(self.last_metrics, f, indent=2)
            print(f'[PCCR] {self.rig_dir}: no predicted box -> mAP7 0 (eval.json written without the devkit)', flush=True)
            return self.last_metrics
        subprocess.run([self.devkit_py, osp.abspath(DEVKIT_SCRIPT), self.rig_dir, osp.abspath(result_path), out_dir,
                        '--version', self.version, '--scene-prefix', ''], check=True)
        with open(osp.join(out_dir, 'eval.json')) as f:
            self.last_metrics = json.load(f)
        return self.last_metrics

    def _format_bbox(self, results, img_metas, jsonfile_prefix=None):
        """DetNuscEvaluator._format_bbox; attribute default '' for PCCR's classes."""
        nusc_annos = {}
        mapped_class_names = self.class_names

        print('Start to convert detection format...')

        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            boxes, scores, labels = det
            boxes = boxes
            sample_token = img_metas[sample_id]['token']
            trans = np.array(img_metas[sample_id]['ego2global_translation'])
            rot = Quaternion(img_metas[sample_id]['ego2global_rotation'])
            annos = list()
            for i, box in enumerate(boxes):
                name = mapped_class_names[labels[i]]
                center = box[:3]
                wlh = box[[4, 3, 5]]
                box_yaw = box[6]
                box_vel = box[7:].tolist()
                box_vel.append(0)
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
                nusc_box = Box(center, wlh, quat, velocity=box_vel)
                nusc_box.rotate(rot)
                nusc_box.translate(trans)
                if np.sqrt(nusc_box.velocity[0]**2 +
                           nusc_box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = self.DefaultAttribute.get(name, '')  # PCCR port
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = self.DefaultAttribute.get(name, '')  # PCCR port
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=nusc_box.center.tolist(),
                    size=nusc_box.wlh.tolist(),
                    rotation=nusc_box.orientation.elements.tolist(),
                    velocity=nusc_box.velocity[:2],
                    detection_name=name,
                    detection_score=float(scores[i]),
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            # other views results of the same frame should be concatenated
            if sample_token in nusc_annos:
                nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }
        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path
