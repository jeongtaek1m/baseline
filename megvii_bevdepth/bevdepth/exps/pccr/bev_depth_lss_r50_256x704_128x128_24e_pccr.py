# Gen2E: the official Megvii BEVDepth (base_exp.py, single frame) on PCCR.
#
# User decisions 2026-11-23 (docs/decisions.md):
#   * base_exp.py as is: R50 + SECONDFPN, DepthNet (ASPP + DCN, camera-aware MLP), depth bins
#     [2, 58, 0.5], BEV 80 ch, ResNet-18 BEV encoder + SECONDFPN, final_dim 256x704, no CBGS,
#     no EMA, AdamW weight decay 1e-7, MultiStepLR [19, 23], 24 epochs, gradient clip 5,
#     velocity head kept (code weights 0.5);
#   * total batch 8 (2 GPUs x 4) with lr 2e-4 (basic_lr_per_img = 2e-4 / 8);
#   * FP32 instead of base_cli's fp16 (pass --precision 32, see tools/megvii_train.sh);
#   * PCCR's head setting: the 9 PCCR classes in one task, rotated NMS 0.2 (pre 1000,
#     post 500, score 0.1) as PCCR's BEVDet config; mAP is reported over 7 classes.
# PCCR adaptations the official code cannot express:
#   * 1280x720 images: resize_lim scaled with the width (0.386-0.55 for 1600 wide = 0.877-1.25x
#     of the test scale 0.44 -> 0.4825-0.6875 for 1280 wide, test scale 0.55);
#   * rig cameras (4 to 10, rig-specific names) read from the rig's info pkl;
#   * PCCR category names added to the dataset's name mapping (they already are the detection
#     names); infos from scripts/gen_info_pccr.py (PCCR's own train/val/test samples);
#   * predictions are written as results_nusc.json and scored with PCCR's evaluation
#     (tools/score_megvii_pccr.py), so the evaluator here only formats boxes.
#
# Environment (set by tools/megvii_train.sh / tools/megvii_test_rigs.sh):
#   PCCR_ROOT      rigs root (default: <repo>/data/pccr)
#   PCCR_INFO_DIR  Megvii-format infos (default: megvii_bevdepth/data/pccr_infos)
#   PCCR_RIG       rig to predict on with -p (default R1); training always uses R1 train
#   PCCR_SPLIT     split to predict on (default test)
#   PCCR_OUT       directory for results_nusc.json of -p (default <default_root_dir>/test_on_<RIG>_<SPLIT>)
import copy
import os

import mmcv

from bevdepth.datasets import nusc_det_dataset
from bevdepth.evaluators.det_evaluators import DetNuscEvaluator
from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.base_exp import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel
from bevdepth.exps.nuscenes.base_exp import head_conf, ida_aug_conf

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
PCCR_ROOT = os.environ.get('PCCR_ROOT', os.path.join(REPO, 'data', 'pccr'))
PCCR_INFO_DIR = os.environ.get('PCCR_INFO_DIR',
                               os.path.join(REPO, 'megvii_bevdepth', 'data', 'pccr_infos'))

PCCR_CLASSES = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child',
                'traffic_sign', 'traffic_light']
# PCCR's category names are already detection names (category.json); the official mapping
# only knows nuScenes' hierarchical names.
nusc_det_dataset.map_name_from_general_to_detection.update({c: c for c in PCCR_CLASSES})

pccr_head_conf = copy.deepcopy(head_conf)
pccr_head_conf['tasks'] = [dict(num_class=len(PCCR_CLASSES), class_names=PCCR_CLASSES)]
pccr_head_conf['test_cfg'].update(nms_type='rotate', nms_thr=0.2, pre_max_size=1000,
                                  post_max_size=500, min_radius=[4])

pccr_ida_aug_conf = copy.deepcopy(ida_aug_conf)
pccr_ida_aug_conf.update(resize_lim=(0.4825, 0.6875), H=720, W=1280)


def info_path(rig, split):
    return os.path.join(PCCR_INFO_DIR, f'{rig}_megvii_infos_{split}.pkl')


def rig_cams(rig, split):
    return list(mmcv.load(info_path(rig, split))[0]['cam_infos'].keys())


class PCCREvaluator(DetNuscEvaluator):
    """Formats predictions as results_nusc.json in `output_dir` (scored by PCCR's evaluation)."""

    # PCCR's adapter (third_party/pccr/models/BEVDet/pccr_dataset.py, DefaultAttribute)
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'bicycle': 'cycle.without_rider',
        'motorcycle': 'cycle.without_rider',
        'adult': 'pedestrian.moving',
        'child': 'pedestrian.moving',
        'traffic_sign': '',
        'traffic_light': '',
    }

    def _format_bbox(self, results, img_metas, jsonfile_prefix=None):
        # base_cli's predict branch passes the checkpoint's folder; write per rig instead
        mmcv.mkdir_or_exist(self.output_dir)
        return super()._format_bbox(results, img_metas, self.output_dir)


class BEVDepthLightningModel(BaseBEVDepthLightningModel):

    def __init__(self, **kwargs):
        predict = bool(kwargs.get('predict'))
        rig = os.environ.get('PCCR_RIG', 'R1') if predict else 'R1'
        split = os.environ.get('PCCR_SPLIT', 'test') if predict else 'train'
        ida = copy.deepcopy(pccr_ida_aug_conf)
        ida['cams'] = rig_cams(rig, split)
        ida['Ncams'] = len(ida['cams'])
        kwargs.update(data_root=os.path.join(PCCR_ROOT, rig), class_names=PCCR_CLASSES,
                      head_conf=pccr_head_conf, ida_aug_conf=ida)
        super().__init__(**kwargs)
        self.basic_lr_per_img = 2e-4 / 8  # lr 2e-4 at total batch 8
        self.train_info_paths = info_path('R1', 'train')
        self.val_info_paths = info_path('R1', 'val')
        self.predict_info_paths = info_path(rig, split)
        out = os.environ.get('PCCR_OUT') or os.path.join(self.default_root_dir,
                                                         f'test_on_{rig}_{split}')
        self.evaluator = PCCREvaluator(class_names=PCCR_CLASSES, output_dir=out)


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel, 'bev_depth_lss_r50_256x704_128x128_24e_pccr')
