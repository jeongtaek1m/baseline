# PD-BEV port (Gen2E): new file. The co-author's PCCR BEVDepth exp (base_exp_pccr.py) + PD-BEV.
"""BEVDepth + PD-BEV (domain generalisation) on PCCR: train R1, test the 14 rigs. Base = the co-author's plain PCCR
BEVDepth (base_exp_pccr.PCCRBEVDepthLightningModel: official Megvii single-frame BEVDepth, 256 x 704, 9 classes in one
task with rotate NMS, lr 2e-4 fixed, AdamW wd 1e-7, MultiStepLR [19, 23], 24 epochs, batch 4 x 2 GPUs, no CBGS, no
EMA), unchanged except what PD-BEV changes (decisions 2026-11-23, docs/decisions.md):

- model: PDBEVBEVDepth (bevdepth/models/pdbev_bev_depth.py): virtual depth with PD-BEV's bins d_bound [1, 100, 1]
  (official DG config; x / y / z bounds stay the base's, z [-5, 3] = official PD-BEV), 31-dim camera input, IFV with
  4 height bins, real + virtual view rendering, Img_Aux / Img_Aux_Dy, box-centre depth loss; channels as official
  PD-BEV (image neck 512, BEV 80, BEV neck 256), which the Megvii base already has;
- data: PCCRPDBEVDataset (actual intrinsics, virtual cameras, 2D targets); no LiDAR depth (PD-BEV does not use it);
- the virtual-camera perturbation ranges: PDBEV_RANGE = 'A' (official) or 'B' (PCCR rig changes), set by the exp;
- the old-base PD-BEV bug fixes PDBEV-1 ... 18 as they apply (docs/bug_ledger.md).

Train in fp32 (decision 2026-11-23): --precision 32. Test with -e --test_rigs (the co-author's PCCR scoring).
"""
import time

import mmcv
import torch
import torch.utils.data

from bevdepth.datasets.pccr_det_dataset import PCCR_CLASSES
from bevdepth.datasets.pccr_pdbev_dataset import PDBEV_AUG, PCCRPDBEVDataset, collate_fn_pdbev
from bevdepth.exps.pccr.base_exp_pccr import PCCRBEVDepthLightningModel, ida_conf_for, rig_dir
from bevdepth.models.pdbev_bev_depth import PDBEVBEVDepth
from bevdepth.utils.torch_dist import get_rank

__all__ = ['PCCRPDBEVLightningModel']

PDBEV_D_BOUND = [1.0, 100.0, 1.0]  # PD-BEV DG config: virtual depth bins


class PCCRPDBEVLightningModel(PCCRBEVDepthLightningModel):
    PDBEV_RANGE = None  # 'A' or 'B' (the exp files)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.PDBEV_RANGE in PDBEV_AUG, 'use an exp that sets PDBEV_RANGE'
        self.pdbev_aug = PDBEV_AUG[self.PDBEV_RANGE]
        self.backbone_conf['d_bound'] = list(PDBEV_D_BOUND)
        self.dbound = self.backbone_conf['d_bound']
        self.depth_channels = int((self.dbound[1] - self.dbound[0]) / self.dbound[2])
        self.model = PDBEVBEVDepth(self.backbone_conf, self.head_conf, class_names=list(self.class_names))
        self.data_return_depth = False  # PD-BEV: no LiDAR depth
        self.print_every = kwargs.get('print_every', 50)
        self._t_last = None
        if get_rank() == 0:
            print(f'[PD-BEV] range {self.PDBEV_RANGE}: {self.pdbev_aug}; d_bound {self.dbound}', flush=True)

    def training_step(self, batch):
        (sweep_imgs, mats, _, _, gt_boxes, gt_labels, targets_2d) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
            gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
            gt_labels = [gt_label.cuda() for gt_label in gt_labels]
            targets_2d = {k: v.cuda() for k, v in targets_2d.items()}
        preds, losses = self.model.forward_train(sweep_imgs, mats, targets_2d)
        targets = self.model.get_targets(gt_boxes, gt_labels)
        detection_loss = self.model.loss(targets, preds)
        self.log('detection_loss', detection_loss)
        for name, value in losses.items():
            self.log(name, value)
        if get_rank() == 0 and self.print_every > 0 and self.global_step % self.print_every == 0:
            now = time.time()
            dt = '' if self._t_last is None else f' {(now - self._t_last) / self.print_every:.3f} s/iter'
            self._t_last = now
            print(f'[PD-BEV] epoch {self.current_epoch} iter {self.global_step} det {float(detection_loss):.4f} ' +
                  ' '.join(f'{k} {float(v):.4f}' for k, v in losses.items()) + dt, flush=True)
        return detection_loss + sum(losses.values())

    def _dataset(self, info_path, rig, is_train, return_depth):
        infos = mmcv.load(info_path)
        conf = ida_conf_for(self.ida_aug_conf, infos)
        return PCCRPDBEVDataset(ida_aug_conf=conf, bda_aug_conf=self.bda_aug_conf, classes=PCCR_CLASSES,
                                data_root=rig_dir(self.pccr_root, rig), info_paths=info_path, is_train=is_train,
                                return_depth=False, use_cbgs=False, img_conf=self.img_conf, num_sweeps=1,
                                sweep_idxes=list(), key_idxes=list(), use_fusion=False,
                                pdbev_aug=self.pdbev_aug if is_train else None)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self._dataset(self.train_info_paths, 'R1', True, False),
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False,
            collate_fn=collate_fn_pdbev,
            sampler=None,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._dataset(self.val_info_paths, self.test_rig, False, False),
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=collate_fn_pdbev,
            num_workers=self.num_workers,
            sampler=None,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()

