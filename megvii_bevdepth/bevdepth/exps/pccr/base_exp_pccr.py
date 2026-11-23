# PCCR port: new file. The official single-frame base_exp (bevdepth/exps/nuscenes/base_exp.py) on PCCR.
"""BEVDepth (official base_exp: R50 + SECONDFPN, camera-aware DepthNet, d_bound [2, 58, 0.5], BEV +-51.2 m at 0.8 m,
z [-5, 3], depth loss x3, single frame, CBGS off, no EMA, 24 epochs AdamW wd 1e-7 MultiStepLR [19, 23], amp, clip 5)
trained on PCCR's R1 and tested on its 14 rigs. Only what PCCR forces differs, each marked # PCCR port:

- data: Megvii-format infos made from each rig's nuScenes tables (scripts/gen_info_pccr.py), PCCRDetDataset (PCCR's
  class names); the cameras and the image size of every dataset come from its infos (R1: 8 cameras, 1280 x 720;
  UniDrive: the virtual cameras, 800 x 450);
- image: final_dim (256, 704), the official one: resize to width ~704, keep the bottom 256 rows (bot_pct_lim (0, 0)),
  train and test, as the official code computes it (switch --final_h 384: PCCR's BEVDet input 384 x 704); resize_lim
  (0.4826, 0.6875) at W = 1280 = the official (0.386, 0.55) relative to its test scale 704/1600, times 704/1280,
  scaled by 1280/W for other widths; test-time resize / crop as the official code computes them;
- classes: PCCR's 9 classes as one task (PCCR's protocol, as every PCCR BEVDepth row). With one task the official
  per-task circle NMS radii (car 4, ..., pedestrian 0.175) cannot be applied per class, so the head uses the
  official code's 'rotate' NMS path (class-agnostic BEV IoU 0.2 within the task, as PCCR's BEVDet and our other
  BEVDepth rows) and post_max_size 500 (the official cap is 83 per task, ~500 over its 6 tasks);
- lr: fixed 2e-4 (PCCR's BEVDet protocol, the user's decision) instead of base_exp's 2e-4 / 64 x batch;
  --lr_official restores the formula; the value used is printed;
- batch 4 per GPU (x 2 GPUs = 8);
- evaluation: PCCRDetEvaluator (official result-file conversion, PCCR's devkit scoring per rig).
"""
import copy
import os
from functools import partial

import mmcv
import torch
import torch.utils.data
from torch.optim.lr_scheduler import MultiStepLR

from bevdepth.datasets.pccr_det_dataset import PCCR_CLASSES, PCCRDetDataset, collate_fn
from bevdepth.evaluators.pccr_det_evaluator import PCCRDetEvaluator
from bevdepth.exps.nuscenes import base_exp as official
from bevdepth.exps.nuscenes.base_exp import BEVDepthLightningModel

TEST_RIGS = ['R1', 'R1-t', 'R1-r', 'R1-f', 'R1-c6', 'R1-c10', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9']
FINAL_DIMS = {256: (256, 704), 384: (384, 704)}  # PCCR port: the official one (default) / PCCR's BEVDet input
RESIZE_LIM_1280 = (0.4826, 0.6875)  # PCCR port: official (0.386, 0.55) / (704 / 1600) * (704 / 1280)
LR = 2e-4  # PCCR port: PCCR's BEVDet lr (official base_exp: 2e-4 / 64 per image)


def rig_dir(root, rig):
    """<root>/<rig> holding v1.0-*/ (Poseidon, NEO) or <root>/<rig>/<rig> (shannon's extracted zips)."""
    d = os.path.join(root, rig)
    return d if os.path.isdir(os.path.join(d, 'v1.0-test')) else os.path.join(d, rig)


def pccr_confs(final_h=256):
    """The official backbone / head / ida / bda confs with PCCR's changes (cameras and H, W are set per dataset)."""
    final_dim = FINAL_DIMS[int(final_h)]
    backbone_conf = copy.deepcopy(official.backbone_conf)
    backbone_conf['final_dim'] = final_dim  # PCCR port
    ida_aug_conf = copy.deepcopy(official.ida_aug_conf)
    ida_aug_conf.update(final_dim=final_dim, H=720, W=1280, resize_lim=RESIZE_LIM_1280, cams=[],
                        Ncams=0)  # PCCR port: cams / H / W / resize_lim are filled per dataset (ida_conf_for)
    head_conf = copy.deepcopy(official.head_conf)
    head_conf['tasks'] = [dict(num_class=len(PCCR_CLASSES), class_names=list(PCCR_CLASSES))]  # PCCR port: one task
    head_conf['test_cfg'].update(nms_type='rotate', nms_thr=0.2, post_max_size=500,
                                 min_radius=[4])  # PCCR port (min_radius is used by 'circle' only)
    bda_aug_conf = copy.deepcopy(official.bda_aug_conf)
    return backbone_conf, head_conf, ida_aug_conf, bda_aug_conf


def ida_conf_for(ida_aug_conf, infos):
    """PCCR port: the dataset's own cameras (in the order of its infos) and image size; resize_lim by the rule above."""
    cams = list(infos[0]['cam_infos'].keys())
    sizes = {(c['width'], c['height']) for c in infos[0]['cam_infos'].values()}
    assert len(sizes) == 1, f'one image size per rig expected, got {sizes}'
    (w, h), = sizes
    conf = copy.deepcopy(ida_aug_conf)
    conf.update(cams=cams, Ncams=len(cams), H=h, W=w,
                resize_lim=tuple(r * 1280.0 / w for r in RESIZE_LIM_1280))
    return conf


def build_pccr_dataset(info_path, data_root, ida_aug_conf, bda_aug_conf, is_train, return_depth, **kwargs):
    """PCCRDetDataset with the cameras / image size of its infos (shared by the exp, the GT gate and the checks)."""
    infos = mmcv.load(info_path)
    conf = ida_conf_for(ida_aug_conf, infos)
    ds = PCCRDetDataset(ida_aug_conf=conf, bda_aug_conf=bda_aug_conf, classes=PCCR_CLASSES, data_root=data_root,
                        info_paths=info_path, is_train=is_train, return_depth=return_depth, **kwargs)
    return ds


class PCCRBEVDepthLightningModel(BEVDepthLightningModel):
    INFO_ROOT = 'data/pccr_megvii'  # variants (UniDrive) override this

    def __init__(self,
                 gpus=1,
                 pccr_root='/data4/pccr/data',
                 info_root=None,
                 batch_size_per_device=4,
                 lr=LR,
                 lr_official=False,
                 final_h=256,
                 test_rig='R1',
                 num_workers=4,
                 default_root_dir='./outputs/',
                 **kwargs):
        backbone_conf, head_conf, ida_aug_conf, bda_aug_conf = pccr_confs(final_h)
        kwargs.pop('data_root', None)
        super().__init__(gpus=gpus, data_root=pccr_root, batch_size_per_device=batch_size_per_device,
                         class_names=list(PCCR_CLASSES), backbone_conf=backbone_conf, head_conf=head_conf,
                         ida_aug_conf=ida_aug_conf, bda_aug_conf=bda_aug_conf, default_root_dir=default_root_dir,
                         **kwargs)
        self.pccr_root = pccr_root
        self.info_root = info_root or self.INFO_ROOT
        self.lr = lr
        self.lr_official = lr_official
        self.num_workers = num_workers  # official: 4
        self.train_info_paths = os.path.join(self.info_root, 'R1_infos_train.pkl')
        self.set_test_rig(test_rig, os.path.join(default_root_dir, 'pccr_test'))

    def set_test_rig(self, rig, out_root):
        """PCCR port: the rig the val / test / predict dataloaders read and its scoring database."""
        self.test_rig = rig
        self.val_info_paths = os.path.join(self.info_root, f'{rig}_infos_test.pkl')
        self.predict_info_paths = self.val_info_paths
        self.evaluator = PCCRDetEvaluator(class_names=self.class_names, output_dir=os.path.join(out_root, rig),
                                          rig_dir=rig_dir(self.pccr_root, rig), version='v1.0-test')

    def configure_optimizers(self):
        if self.lr_official:  # base_exp's formula; world size from the trainer (official: self.gpus)
            lr = self.basic_lr_per_img * self.batch_size_per_device * self.trainer.world_size
        else:
            lr = self.lr  # PCCR port
        print(f'[PCCR] lr = {lr:.3e} ({"2e-4/64 x batch" if self.lr_official else "fixed"}), batch '
              f'{self.batch_size_per_device} x {self.trainer.world_size} GPUs, AdamW wd 1e-7, MultiStepLR [19, 23]',
              flush=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-7)
        scheduler = MultiStepLR(optimizer, [19, 23])
        return [[optimizer], [scheduler]]

    def _dataset(self, info_path, rig, is_train, return_depth):  # PCCR port
        return build_pccr_dataset(info_path, rig_dir(self.pccr_root, rig), self.ida_aug_conf, self.bda_aug_conf,
                                  is_train=is_train, return_depth=return_depth, use_cbgs=self.data_use_cbgs and is_train,
                                  img_conf=self.img_conf, num_sweeps=self.num_sweeps, sweep_idxes=self.sweep_idxes,
                                  key_idxes=self.key_idxes, use_fusion=self.use_fusion)

    def train_dataloader(self):
        train_dataset = self._dataset(self.train_info_paths, 'R1', True, self.data_return_depth)  # PCCR port
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(collate_fn, is_return_depth=self.data_return_depth or self.use_fusion),
            sampler=None,
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = self._dataset(self.val_info_paths, self.test_rig, False, self.use_fusion)  # PCCR port
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=partial(collate_fn, is_return_depth=self.use_fusion),
            num_workers=self.num_workers,
            sampler=None,
        )
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()  # PCCR port: the test split of the chosen rig

    @staticmethod
    def add_model_specific_args(parent_parser):  # PCCR port
        parser = parent_parser
        parser.add_argument('--pccr_root', default='/data4/pccr/data', help='raw PCCR rigs (<root>/<rig>/v1.0-*)')
        parser.add_argument('--info_root', default=None, help='Megvii-format infos (default: the exp\'s)')
        parser.add_argument('--lr', type=float, default=LR, help='fixed lr (PCCR protocol: 2e-4)')
        parser.add_argument('--lr_official', action='store_true', help="base_exp's 2e-4 / 64 x total batch")
        parser.add_argument('--final_h', type=int, default=256, choices=sorted(FINAL_DIMS),
                            help='final_dim (final_h, 704): 256 = official (default), 384 = PCCR BEVDet input')
        parser.add_argument('--num_workers', type=int, default=4, help='dataloader workers per GPU (official 4)')
        parser.add_argument('--test_rigs', nargs='+', default=None,
                            help='with -e: score these PCCR test rigs one after another (e.g. all 14)')
        parser.add_argument('--eval_out', default=None, help='with -e: <eval_out>/<rig>/{results_nusc,eval}.json')
        parser.add_argument('--print_every', type=int, default=50, help='print losses / speed every n iterations')
        return parser
