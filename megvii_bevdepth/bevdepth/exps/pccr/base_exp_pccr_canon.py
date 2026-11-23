# PCCR port: new file. Our canonical-camera method on the Megvii BEVDepth PCCR exp (port of the BEVDet configs
# pccr-canon-bevdepth-1f.py / pccr-canon-bevdepth-vs-1f.py and mmdet3d/canon of CoIn3D/BEVDet).
"""Same model, schedule and data as the plain PCCR exp (base_exp_pccr: official base_exp, single frame, no EMA, no CBGS,
24 epochs, AdamW 2e-4 wd 1e-7, MultiStepLR [19, 23], batch 4 x 2, amp, clip 5, rotate NMS), with the inputs
normalised to the levelled cylindrical camera of fixed angular resolution:

- PCCRCanonDetDataset (bevdepth/datasets/pccr_canon_dataset.py): every camera resampled to the canonical camera
  (rho 552.75 px/rad, pitch / roll levelled, layout 'strip', crop to valid, padded to /32); the official ida
  resize / crop / rotate dropped; per-view horizontal flip kept (exact, in ida_mats); optional vertical shift
  (canon_vs: 20 px, training only); depth labels = LiDAR horizontal range;
- CanonBEVDepth / CanonLSSFPN (bevdepth/layers/backbones/canon_lss_fpn.py): frustum from the canonical rays, depth
  bins (official d_bound [2, 58, 0.5]) along the horizontal range, invalid cells not lifted, camera-aware DepthNet fed
  the canonical intrinsics (rho, rho, 0, 0) and the levelled pose;
- elevation band (rule of this port): computed from R1's own calibration as the part of R1 the plain row sees through
  the official test crop of the configured final_dim (default (256, 704): -19.71 .. +6.00 deg, 256 rows at rho);
  --canon_band full: R1's whole vertical field of view (+-19.71 deg). Test rigs use the same band (R1 information
  only).
"""
import math
from functools import partial

import mmcv
import torch
import torch.utils.data

from bevdepth.datasets.pccr_canon_dataset import (DEFAULT_CANON, PCCRCanonDetDataset, canon_collate_fn,
                                                  plain_row_band_deg)
from bevdepth.datasets.pccr_det_dataset import PCCR_CLASSES
from bevdepth.exps.pccr.base_exp_pccr import PCCRBEVDepthLightningModel, ida_conf_for, rig_dir
from bevdepth.models.canon_bev_depth import CanonBEVDepth

RHO = 552.75


def make_canon_conf(train_info_path, ida_aug_conf, band='plain', rho=RHO, vshift=0, prefilter=True, verbose=True):
    """canon_conf of PCCRCanonDetDataset: the elevation band from R1's calibration (first training sample) and the
    official test crop of ida_aug_conf['final_dim'] (band 'plain') or R1's full image (band 'full')."""
    infos = mmcv.load(train_info_path)
    lo, hi = plain_row_band_deg(infos[0]['cam_infos'], ida_aug_conf['final_dim'], ida_aug_conf['bot_pct_lim'],
                                full=band == 'full')
    rows = int(round(rho * (math.tan(math.radians(hi)) - math.tan(math.radians(lo)))))
    conf = dict(DEFAULT_CANON, rho=float(rho), elev_range_deg=(lo, hi), max_rows=rows, vshift=int(vshift),
                prefilter=bool(prefilter))
    if verbose:
        print(f'[PCCR canon] rho {rho} px/rad, band {band}: elevation {lo:.3f} .. {hi:.3f} deg (R1, final_dim '
              f'{tuple(ida_aug_conf["final_dim"])}), {rows} rows; vshift {vshift} px (train), prefilter {prefilter}',
              flush=True)
    return conf


def build_pccr_canon_dataset(info_path, data_root, ida_aug_conf, bda_aug_conf, canon_conf, is_train, return_depth,
                             **kwargs):
    infos = mmcv.load(info_path)
    conf = ida_conf_for(ida_aug_conf, infos)  # the rig's cameras (H / W / resize_lim unused by the canonical path)
    return PCCRCanonDetDataset(ida_aug_conf=conf, bda_aug_conf=bda_aug_conf, classes=PCCR_CLASSES,
                               data_root=data_root, info_paths=info_path, is_train=is_train,
                               return_depth=return_depth, canon_conf=canon_conf, **kwargs)


class PCCRCanonBEVDepthLightningModel(PCCRBEVDepthLightningModel):
    VSHIFT = 0  # canon_vs: 20

    def __init__(self, canon_band='plain', canon_vshift=None, canon_rho=RHO, canon_prefilter=1,
                 mlp_principal_point=False, **kwargs):
        super().__init__(**kwargs)
        vshift = self.VSHIFT if canon_vshift is None else canon_vshift
        self.canon_conf = make_canon_conf(self.train_info_paths, self.ida_aug_conf, band=canon_band, rho=canon_rho,
                                          vshift=vshift, prefilter=bool(canon_prefilter))
        self.model = CanonBEVDepth(self.backbone_conf, self.head_conf, is_train_depth=True,
                                   canon_backbone_conf=dict(mlp_principal_point=mlp_principal_point))

    def _dataset(self, info_path, rig, is_train, return_depth):
        return build_pccr_canon_dataset(info_path, rig_dir(self.pccr_root, rig), self.ida_aug_conf, self.bda_aug_conf,
                                        self.canon_conf, is_train=is_train, return_depth=return_depth,
                                        use_cbgs=self.data_use_cbgs and is_train, img_conf=self.img_conf,
                                        num_sweeps=self.num_sweeps, sweep_idxes=self.sweep_idxes,
                                        key_idxes=self.key_idxes, use_fusion=self.use_fusion)

    def train_dataloader(self):  # base_exp_pccr's, with the canonical collate_fn
        train_dataset = self._dataset(self.train_info_paths, 'R1', True, self.data_return_depth)
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(canon_collate_fn, is_return_depth=self.data_return_depth or self.use_fusion),
            sampler=None,
        )

    def val_dataloader(self):
        val_dataset = self._dataset(self.val_info_paths, self.test_rig, False, self.use_fusion)
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=partial(canon_collate_fn, is_return_depth=self.use_fusion),
            num_workers=self.num_workers,
            sampler=None,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = PCCRBEVDepthLightningModel.add_model_specific_args(parent_parser)
        parser.add_argument('--canon_band', default='plain', choices=['plain', 'full'],
                            help="elevation band: 'plain' = R1's region the plain row sees (official crop of "
                                 "final_dim), 'full' = R1's whole vertical FOV")
        parser.add_argument('--canon_vshift', type=int, default=None, help='training vertical shift (px); exp default')
        parser.add_argument('--canon_rho', type=float, default=RHO, help='canonical angular resolution (px/rad)')
        parser.add_argument('--canon_prefilter', type=int, default=1, choices=[0, 1],
                            help='PIL-downsample each photo by min(1, rho/f) before the cylindrical remap')
        parser.add_argument('--mlp_principal_point', action='store_true',
                            help='feed (cx, cy) to the camera-aware DepthNet (ablation; default 0, 0)')
        return parser
