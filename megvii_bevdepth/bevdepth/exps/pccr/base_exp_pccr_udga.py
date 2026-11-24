# UDGA port (Gen2E): new file. The co-author's PCCR BEVDepth exp (base_exp_pccr.py) + UDGA.
# UDGA is Samsung-proprietary; this is a re-implementation and stays in the private repository.
"""BEVDepth + UDGA (domain generalisation) on PCCR: train R1, test the 14 rigs. Base = the co-author's plain PCCR
BEVDepth (base_exp_pccr.PCCRBEVDepthLightningModel), unchanged except what UDGA changes (decisions 2026-11-24,
docs/decisions.md):

- model: UDGABEVDepth (bevdepth/models/udga_bev_depth.py): DG-BEV virtual depth, pixel_const 400, bins
  d_bound [1, 90, 1] (UDGA config); L_ov weight 1; L_p weight 2 ('public' or 'paper' by the exp);
- data: PCCRUDGADataset (ext aug: +{-90, 90, 180} deg on the bda rotation with probability 0.5, training only);
- LiDAR depth (the exp's LIDAR_DEPTH): on = the base's own depth loss (softmax BCE, x3) on virtual targets d * k
  (UDGA-Ld, the public config's composition); off = no LiDAR depth (UDGA-Lp, the paper's final row);
- 24 epochs and the rest of the base schedule (decision 2026-11-24; the public config has 32).

Train in fp32: --precision 32. Test with -e --test_rigs (the co-author's PCCR scoring).
"""
import time

import mmcv
import torch

from bevdepth.datasets.pccr_det_dataset import PCCR_CLASSES
from bevdepth.datasets.pccr_udga_dataset import PCCRUDGADataset
from bevdepth.exps.pccr.base_exp_pccr import PCCRBEVDepthLightningModel, ida_conf_for, rig_dir
from bevdepth.models.udga_bev_depth import UDGABEVDepth
from bevdepth.utils.torch_dist import get_rank

__all__ = ['PCCRUDGALightningModel']

UDGA_D_BOUND = [1.0, 90.0, 1.0]  # UDGA config: virtual depth bins (pixel_const 400)


class PCCRUDGALightningModel(PCCRBEVDepthLightningModel):
    PHOTO_MODE = None  # 'public' or 'paper' (the exp files)
    LIDAR_DEPTH = None  # True (UDGA-Ld) or False (UDGA-Lp)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.PHOTO_MODE in ('public', 'paper') and self.LIDAR_DEPTH in (True, False), \
            'use an exp that sets PHOTO_MODE and LIDAR_DEPTH'
        self.backbone_conf['d_bound'] = list(UDGA_D_BOUND)
        self.dbound = self.backbone_conf['d_bound']
        self.depth_channels = int((self.dbound[1] - self.dbound[0]) / self.dbound[2])
        self.model = UDGABEVDepth(self.backbone_conf, self.head_conf, pixel_const=400.0, overlap_loss_weight=1.0,
                                  overlap_loss_beta=1.0, photo_loss_weight=2.0, photo_loss_mode=self.PHOTO_MODE)
        self.data_return_depth = self.LIDAR_DEPTH
        self.print_every = kwargs.get('print_every', 50)
        self._t_last = None
        if get_rank() == 0:
            print(f'[UDGA] L_p {self.PHOTO_MODE}, LiDAR depth {self.LIDAR_DEPTH}, d_bound {self.dbound}, '
                  f'pixel_const 400, ext aug p 0.5', flush=True)

    def training_step(self, batch):
        if self.LIDAR_DEPTH:
            (sweep_imgs, mats, _, _, gt_boxes, gt_labels, depth_labels) = batch
        else:
            (sweep_imgs, mats, _, _, gt_boxes, gt_labels) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
            gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
            gt_labels = [gt_label.cuda() for gt_label in gt_labels]
        preds, depth_logits, losses, corr = self.model.forward_train(sweep_imgs, mats)
        targets = self.model.get_targets(gt_boxes, gt_labels)
        detection_loss = self.model.loss(targets, preds)
        if self.LIDAR_DEPTH:
            if len(depth_labels.shape) == 5:  # only the key frame has depth labels (official)
                depth_labels = depth_labels[:, 0, ...]
            # UDGA-Ld: the base's depth loss with the LiDAR depth moved into the virtual bins, v = d * k (DG-BEV eq. 3;
            # UDGA-2: the official target d / k is lifted to d / k^2). k is constant per camera, so it commutes with
            # the min-pooling of get_downsampled_gt_depth.
            k = self.model.backbone.depth_scale(mats['intrin_mats'][:, 0], mats['ida_mats'][:, 0])
            losses['depth_loss'] = self.get_depth_loss(depth_labels.cuda() * k[..., None, None],
                                                       depth_logits.softmax(1))
        self.log('detection_loss', detection_loss)
        for name, value in losses.items():
            self.log(name, value)
        if get_rank() == 0 and self.print_every > 0 and self.global_step % self.print_every == 0:
            now = time.time()
            dt = '' if self._t_last is None else f' {(now - self._t_last) / self.print_every:.3f} s/iter'
            self._t_last = now
            n_pairs = sum(len(p) for p in corr['pairs']) / max(1, len(corr['pairs']))
            print(f'[UDGA] epoch {self.current_epoch} iter {self.global_step} det {float(detection_loss):.4f} ' +
                  ' '.join(f'{k} {float(v):.4f}' for k, v in losses.items()) +
                  f' pairs/sample {n_pairs:.1f} corr {corr["src"].numel()}' + dt, flush=True)
        return detection_loss + sum(losses.values())

    def _dataset(self, info_path, rig, is_train, return_depth):
        infos = mmcv.load(info_path)
        conf = ida_conf_for(self.ida_aug_conf, infos)
        return PCCRUDGADataset(ida_aug_conf=conf, bda_aug_conf=self.bda_aug_conf, classes=PCCR_CLASSES,
                               data_root=rig_dir(self.pccr_root, rig), info_paths=info_path, is_train=is_train,
                               return_depth=return_depth, use_cbgs=False, img_conf=self.img_conf, num_sweeps=1,
                               sweep_idxes=list(), key_idxes=list(), use_fusion=False,
                               ext_aug_prob=0.5 if is_train else 0.0)
