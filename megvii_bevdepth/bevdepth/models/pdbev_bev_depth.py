# PD-BEV port (Gen2E): new file. BaseBEVDepth + PD-BEV (domain generalisation) on the Megvii base.
"""PD-BEV's DG training (official PCBEV_DG, bevdepth/mmdet3d/models/detectors/pcbev.py, with docs/bug_ledger.md
fixes) on the Megvii BEVDepth model:

- backbone: PDBEVLSSFPN (virtual depth, 31-dim camera input); head: BEVDepthHead (BEV ResNet + SECONDFPN + CenterHead);
- training only: Img_Aux on the image features (L_ps, real-view targets, virtual centre depth); the IFV (BEV_Aux) of
  the BEV-neck output, computed once per step (PDBEV-11), rendered from the real cameras and from the perturbed
  virtual cameras (official get_bev_coor / get_feature_Voxel2Frutums / nerf, IFV read as (B, C, Z, Y, X): PDBEV-1);
  Img_Aux_Dy on both rendered views (L_render, metric centre depth); the box-centre depth loss L_pg on the depth
  distribution (official get_depth_loss with downsample_from_ann 2; no LiDAR depth). No car-only BEV loss (not in the
  paper, decision 2026-11-22). Loss weights as official (all 1, depth 3);
- inference: the plain BEVDepth path (PDBEV-15), i.e. forward() in eval mode returns the head's predictions.
"""
import torch
import torch.nn.functional as F
from mmdet3d.models.dense_heads.centerpoint_head import CenterHead
from torch import nn
from torch.cuda.amp import autocast

from bevdepth.layers.backbones.pdbev_lss_fpn import PDBEVLSSFPN
from bevdepth.layers.heads.bev_depth_head import BEVDepthHead
from bevdepth.layers.heads.pdbev_aux import BEV_Aux, Img_Aux, Img_Aux_Dy

__all__ = ['PDBEVBEVDepth']


class PDBEVHead(BEVDepthHead):

    @autocast(False)
    def forward_with_bev(self, x):
        """BEVDepthHead.forward that also returns the BEV-neck output (PD-BEV's 'bev_encoder' output, the IFV input)."""
        x = x.float()
        trunk_outs = [x]
        if self.trunk.deep_stem:
            x = self.trunk.stem(x)
        else:
            x = self.trunk.conv1(x)
            x = self.trunk.norm1(x)
            x = self.trunk.relu(x)
        for i, layer_name in enumerate(self.trunk.res_layers):
            res_layer = getattr(self.trunk, layer_name)
            x = res_layer(x)
            if i in self.trunk.out_indices:
                trunk_outs.append(x)
        fpn_output = self.neck(trunk_outs)
        return CenterHead.forward(self, fpn_output), fpn_output[0]


class PDBEVBEVDepth(nn.Module):

    def __init__(self, backbone_conf, head_conf, class_names, height_num=4, render_bins=(1, 52),
                 depth_loss_weight=3.0, downsample_from_ann=2):
        super().__init__()
        self.backbone = PDBEVLSSFPN(**backbone_conf)
        self.head = PDBEVHead(**head_conf)
        aux_channels = len(class_names) * 7  # heatmap + 6 regression values per class (official)
        img_c = sum(backbone_conf['img_neck_conf']['out_channels'])  # 512 (official PD-BEV: 512)
        bev_c = backbone_conf['output_channels']  # 80 (official numC_Trans: 80)
        bev_neck_c = sum(head_conf['bev_neck_conf']['out_channels'])  # 256 (official FPN_LSS: 256)
        # official DG config (configs/PDBEV/pdbev-r50-cbgs-NUS2X-dg.py)
        self.img_aux = Img_Aux(numC_input=img_c, class_name=class_names, upsample=[1, 2, 1],
                               num_channels=[256, 128, aux_channels])
        self.bev_img_aux = Img_Aux_Dy(numC_input=bev_c, class_name=class_names, num_layer=[2, 1, 1],
                                      upsample=[2, 1, 1], num_channels=[256, 128, aux_channels])
        self.bev_aux = BEV_Aux(numC_input=bev_neck_c, class_name=class_names, num_layer=[2, 2, 2],
                               height_num=height_num, num_channels=[bev_c, bev_c, bev_c + height_num])
        self.height_num = height_num
        z0, z1 = backbone_conf['z_bound'][:2]
        self.revised_interval = float(z1 - z0) / height_num  # PDBEV-8: float IFV z-bin size
        self.render_bins = render_bins  # official nerf(): sum of depth bins 1 .. 51 (depth_range=52)
        self.depth_loss_weight = depth_loss_weight
        self.downsample_from_ann = downsample_from_ann
        self.d_bound = backbone_conf['d_bound']
        self.depth_channels = self.backbone.depth_channels

    def forward(self, x, mats_dict, timestamps=None):
        """Inference (and the plain training call): the BEVDepth path only (PDBEV-15)."""
        return self.head(self.backbone(x, mats_dict, timestamps))

    def forward_train(self, x, mats_dict, targets_2d):
        """Official PCBEV_DG.extract_img_feat + forward_train: head predictions and the PD-BEV losses."""
        out = self.backbone(x, mats_dict, return_pdbev=True)
        preds, bev_neck = self.head.forward_with_bev(out['bev'])
        # L_ps: 2D head on the image features (targets of the real views, virtual centre depth)
        loss_heat_img, loss_reg_img, _ = self.img_aux(out['img_feats'], targets_2d, 'img', depth_flag='vitual')
        # IFV once per step (PDBEV-11), rendered from the real and from the virtual cameras
        ifv = self.bev_aux(bev_neck)
        pv_real = self.render(ifv, out['geom'])
        virtual_s2e = mats_dict['virtual_sensor2ego_mats'][:, 0]
        geom_virtual = self.backbone.get_geometry(virtual_s2e, mats_dict['intrin_mats'][:, 0],
                                                  mats_dict['ida_mats'][:, 0], mats_dict['bda_mat'],
                                                  mats_dict['actual_intrin_mats'][:, 0])
        pv_virtual = self.render(ifv, geom_virtual)
        mlp_virtual = self.backbone.get_mlp_input(mats_dict, sensor2ego=virtual_s2e)
        # L_render: camera-aware 2D head on the rendered views (metric centre depth)
        loss_heat_pv, loss_reg_pv, _ = self.bev_img_aux(pv_real, out['mlp_input'], targets_2d, 'img',
                                                         depth_flag='real')
        loss_heat_pv_aug, loss_reg_pv_aug, _ = self.bev_img_aux(pv_virtual, mlp_virtual, targets_2d, 'img_aug',
                                                                 depth_flag='real')
        # L_pg: box-centre virtual depth supervises the depth distribution
        loss_depth = self.get_depth_loss(targets_2d['ann_maps_2d'], out['depth'])
        losses = dict(depth_loss=loss_depth,
                      loss_heatmaps_img=loss_heat_img, loss_regess_img=loss_reg_img,
                      loss_heatmaps_pv=loss_heat_pv, loss_regess_pv=loss_reg_pv,
                      loss_heatmaps_pv_aug=loss_heat_pv_aug, loss_regess_pv_aug=loss_reg_pv_aug)
        return preds, losses

    def render(self, ifv, geom):
        """Official get_bev_coor + get_feature_Voxel2Frutums + nerf: the IFV value at every frustum point (index by
        truncation, as official), summed along each ray over depth bins render_bins. Only those bins are gathered
        (the official code gathered all and summed these), so the result is the same.
        ifv (B, C, Z, Y, X), geom (B, N, D, H, W, 3) -> (B * N, C, H, W)."""
        lower = self.backbone.voxel_coord - self.backbone.voxel_size / 2.0
        interval = torch.stack([self.backbone.voxel_size[0], self.backbone.voxel_size[1],
                                self.backbone.voxel_size.new_tensor(self.revised_interval)])
        b0, b1 = self.render_bins
        coor = ((geom[:, :, b0:b1] - lower) / interval).long()
        B, C, Z, Y, X = ifv.shape
        _, N, D, H, W, _ = coor.shape
        out = ifv.new_zeros(B, C, N * D * H * W)
        for b in range(B):
            xi = coor[b, ..., 0].reshape(-1)
            yi = coor[b, ..., 1].reshape(-1)
            zi = coor[b, ..., 2].reshape(-1)
            mask = (xi >= 0) & (xi < X) & (yi >= 0) & (yi < Y) & (zi >= 0) & (zi < self.height_num)
            out[b, :, mask] = ifv[b, :, zi[mask], yi[mask], xi[mask]]
        out = out.view(B, C, N, D, H, W).sum(dim=3)
        return out.permute(0, 2, 1, 3, 4).reshape(B * N, C, H, W)

    def get_depth_loss(self, ann_maps_2d, depth_preds):
        """Official LSSViewTransformer_pc.get_depth_loss: BCE on the bins of the box-centre virtual depths."""
        depth_labels = ann_maps_2d[:, :, :, :, 2, :].max(dim=-1).values  # (B, N, H/8, W/8)
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(depth_preds.float(), depth_labels,
                                                reduction='none').sum() / max(1.0, fg_mask.sum())
        return self.depth_loss_weight * depth_loss

    def get_downsampled_gt_depth(self, gt_depths):
        """Official get_downsampled_gt_depth with downsample_from_ann: [B, N, H, W] -> [B*N*h*w, D] one-hot."""
        downsample = self.downsample_from_ann
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // downsample, downsample, W // downsample, downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, downsample * downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // downsample, W // downsample)
        gt_depths = (gt_depths - (self.d_bound[0] - self.d_bound[2])) / self.d_bound[2]
        gt_depths = torch.where((gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0), gt_depths,
                                torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.depth_channels + 1).view(
            -1, self.depth_channels + 1)[:, 1:]
        return gt_depths.float()

    def get_targets(self, gt_boxes, gt_labels):
        return self.head.get_targets(gt_boxes, gt_labels)

    def loss(self, targets, preds_dicts):
        return self.head.loss(targets, preds_dicts)

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)
