# PD-BEV port (Gen2E): new file. PD-BEV's auxiliary heads on the Megvii BEVDepth base.
"""Img_Aux (2D head on the image features, L_ps), Img_Aux_Dy (camera-aware 2D head on the rendered views, L_render)
and BEV_Aux (the implicit foreground volume, IFV) of the official PD-BEV code, ported from our fixed copy
(bevdepth/mmdet3d/models/necks/img_aux.py, PD-BEV c2995d1 + docs/bug_ledger.md fixes):

- PDBEV-12: IFV = sigmoid(F_height) * F_bev (paper eq. 3);
- PDBEV-17: a 1x1 linear output conv after the last BN + ReLU stage, initialised as the identity;
- PDBEV-13 / PDBEV-14 kept as official (decision 2026-11-22): the focal loss divides its negative term by the number
  of negative pixels, and the regression loss is averaged over all pixels. The official detector returned the
  regression loss as a (B*N, H, W) map that mmdet's parse_losses averaged; here `forward` returns that mean.

Only the DG path is ported: the UDA-only methods (pseudo labels, tensor_split, get_heatmaps) are left out, and the
zero-weighted 'unused' losses are not needed (Lightning's DDP finds unused parameters).
"""
import torch
import torch.utils.checkpoint as checkpoint
from mmdet.models.backbones.resnet import BasicBlock
from torch import nn

from bevdepth.layers.backbones.base_lss_fpn import Mlp, SELayer

__all__ = ['Img_Aux', 'Img_Aux_Dy', 'BEV_Aux', 'gaussian_focal_loss']


def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0):
    """PD-BEV's focal loss for Gaussian targets (official: img_aux.py gaussian_focal_loss, PDBEV-13 kept)."""
    eps = 1e-12
    pos_weights = gaussian_target.eq(1)
    pos_temp = pos_weights.reshape(-1)
    pos_num = len(pos_temp[pos_temp > 0.9])
    neg_num = len(pos_temp) - pos_num
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    pos_loss = pos_loss.sum() / (pos_num + 0.1)
    neg_loss = neg_loss.sum() / (neg_num + 0.1)
    return pos_loss + neg_loss


def _out_conv(channels):
    """PDBEV-17: linear 1x1 output layer, identity at initialisation (the official output equals ReLU(.) >= 0)."""
    conv = nn.Conv2d(channels, channels, kernel_size=1)
    with torch.no_grad():
        conv.weight.copy_(torch.eye(channels).view(channels, channels, 1, 1))
        conv.bias.zero_()
    return conv


def _basic_stages(numC_input, num_layer, num_channels, stride, upsample, norm_cfg):
    """The official 'Basic' stages of Img_Aux / Img_Aux_Dy: BasicBlock (conv downsample), ConvTranspose + BN + ReLU,
    then num_layer - 1 BasicBlocks."""
    layers = []
    curr_numC = numC_input
    for i in range(len(num_layer)):
        layer = [
            BasicBlock(curr_numC, num_channels[i], stride=stride[i],
                       downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1), norm_cfg=norm_cfg)
        ]
        curr_numC = num_channels[i]
        layer.extend([
            nn.ConvTranspose2d(curr_numC, curr_numC, kernel_size=upsample[i], stride=upsample[i], padding=0),
            nn.BatchNorm2d(curr_numC),
            nn.ReLU(inplace=True),
        ])
        layer.extend([BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg) for _ in range(num_layer[i] - 1)])
        layers.append(nn.Sequential(*layer))
    return nn.Sequential(*layers)


class _Aux2DLosses(nn.Module):
    """The official 2D-head losses (Img_Aux.get_target / get_cls_loss / get_regess_loss)."""

    def _init_losses(self, class_name):
        self.class_num = len(class_name)
        self.task_loss_weight = [1.0] * 20

    @staticmethod
    def get_target(flag, targets):
        if flag == 'img':
            return targets['heatmaps_2d'], targets['ann_maps_2d'], targets['heatmap_masks_2d']
        assert flag == 'img_aug'
        return targets['heatmaps_2d_aug'], targets['ann_maps_2d_aug'], targets['heatmap_masks_2d_aug']

    def get_cls_loss(self, pre, gts):
        if gts.dim() == 4:
            gts = gts.unsqueeze(0)
        if gts.dim() == 6:
            gts = gts.squeeze(0)
        B, N, H, W, C = gts.shape
        gts = gts.reshape(-1, H, W, C)
        loss_heatmaps = 0
        for task_id in range(self.class_num):
            loss_heatmap = gaussian_focal_loss(pre[:, task_id, :, :], gts[:, :, :, task_id])
            loss_heatmaps = loss_heatmaps + self.task_loss_weight[task_id] * loss_heatmap
        return loss_heatmaps

    def get_regess_loss(self, pre, gts, masks, depth_flag='vitual'):
        """Per-pixel map (B*N, H, W), as the official code returns it."""
        if gts.dim() == 5:
            gts = gts.unsqueeze(0)
            masks = masks.unsqueeze(0)
        if gts.dim() == 7:
            gts = gts.squeeze(0)
            masks = masks.squeeze(0)
        B, N, H, W, N_ann, C = gts.shape
        gts = gts.reshape(-1, H, W, N_ann, C)
        masks = masks.reshape(-1, H, W, 1, C).repeat(1, 1, 1, 3, 1)
        pre = pre.reshape(B * N, 6, C, H, W).permute(0, 3, 4, 1, 2)
        losses = 0
        for task_id in range(self.class_num):
            loss_uv = torch.mul(masks[..., 0:2, task_id], torch.abs(pre[..., 0:2, task_id] - gts[..., 0:2, task_id]))
            loss_dim = torch.mul(masks[..., 0:3, task_id], torch.abs(pre[..., 3:6, task_id] - gts[..., 3:6, task_id]))
            if depth_flag == 'vitual':  # official spelling: the virtual centre depth (channel 2)
                loss_d = torch.mul(masks[..., 0, task_id], torch.abs(pre[..., 2, task_id] - gts[..., 2, task_id]))
            else:  # 'real': the metric centre depth (channel 6)
                loss_d = torch.mul(masks[..., 0, task_id], torch.abs(pre[..., 2, task_id] - gts[..., 6, task_id]))
            loss = loss_uv.mean(dim=-1) + loss_d + loss_dim.mean(dim=-1)
            losses = losses + self.task_loss_weight[task_id] * loss / 3
        return losses

    def _losses(self, feats, flag, depth_flag, targets):
        heatmaps_2d, ann_maps_2d, heatmap_masks_2d = self.get_target(flag, targets)
        heatmaps_pre = torch.sigmoid(feats[:, 0:self.class_num, ...])
        loss_heatmaps = self.get_cls_loss(heatmaps_pre, heatmaps_2d)
        loss_regess = self.get_regess_loss(feats[:, self.class_num:, ...], ann_maps_2d, heatmap_masks_2d, depth_flag)
        # official: mmdet parse_losses took loss.mean() of the per-pixel map (PDBEV-14 kept)
        return loss_heatmaps, loss_regess.mean(), heatmaps_pre

    def _run_layers(self, x_tmp):
        feats = []
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return self.out_conv(feats[-1])  # PDBEV-17


class Img_Aux(_Aux2DLosses):
    """2D head on the image features (official Img_Aux, 'Basic' blocks); targets of the real views."""

    def __init__(self, numC_input, class_name, num_layer=[2, 2, 2], num_channels=None, stride=[1, 1, 1],
                 upsample=[1, 1, 1], backbone_output_ids=None, norm_cfg=dict(type='BN'), with_cp=False):
        super().__init__()
        assert len(num_layer) == len(stride)
        self._init_losses(class_name)
        num_channels = [numC_input * 2**(i + 1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) if backbone_output_ids is None else backbone_output_ids
        self.layers = _basic_stages(numC_input, num_layer, num_channels, stride, upsample, norm_cfg)
        self.out_conv = _out_conv(num_channels[-1])
        self.with_cp = with_cp

    def forward(self, x, targets, flag='img', depth_flag='vitual'):
        if x.dim() == 5:
            B, Cam_Num, C, H, W = x.shape
            x = x.reshape(-1, C, H, W)
        feats = self._run_layers(x)
        return self._losses(feats, flag, depth_flag, targets)


class Img_Aux_Dy(_Aux2DLosses):
    """Camera-aware 2D head on the rendered views (official Img_Aux_Dy): SE on the 31-dim camera input."""

    def __init__(self, numC_input, class_name, num_layer=[2, 2, 2], num_channels=None, stride=[1, 1, 1],
                 upsample=[1, 1, 1], backbone_output_ids=None, norm_cfg=dict(type='BN'), with_cp=False,
                 mlp_channels=31):
        super().__init__()
        assert len(num_layer) == len(stride)
        self._init_losses(class_name)
        self.bn = nn.BatchNorm1d(mlp_channels)
        self.Pos_mlp = Mlp(mlp_channels, numC_input, numC_input)
        self.Pos_se = SELayer(numC_input)  # NOTE: add camera-aware
        num_channels = [numC_input * 2**(i + 1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) if backbone_output_ids is None else backbone_output_ids
        self.layers = _basic_stages(numC_input, num_layer, num_channels, stride, upsample, norm_cfg)
        self.out_conv = _out_conv(num_channels[-1])
        self.with_cp = with_cp

    def forward(self, x, mlp_input, targets, flag='img', depth_flag='vitual'):
        if x.dim() == 5:
            B, Cam_Num, C, H, W = x.shape
            x = x.reshape(-1, C, H, W)
        mlp_input = mlp_input.squeeze(-1)
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        context_mpl = self.Pos_mlp(mlp_input)[..., None, None]
        x_tmp = self.Pos_se(x, context_mpl)
        feats = self._run_layers(x_tmp)
        return self._losses(feats, flag, depth_flag, targets)


class BEV_Aux(nn.Module):
    """Implicit foreground volume from the BEV features (official BEV_Aux, 'Basic' blocks; PDBEV-12, PDBEV-17)."""

    def __init__(self, numC_input, class_name, num_layer=[2, 2, 2], num_channels=None, stride=[1, 1, 1],
                 height_num=1, backbone_output_ids=None, norm_cfg=dict(type='BN'), with_cp=False):
        super().__init__()
        assert len(num_layer) == len(stride)
        self.class_num = len(class_name)
        self.height_num = height_num
        num_channels = [numC_input * 2**(i + 1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) if backbone_output_ids is None else backbone_output_ids
        features = []
        curr_numC = numC_input
        for i in range(len(num_layer)):
            layer = [
                BasicBlock(curr_numC, num_channels[i], stride=stride[i],
                           downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1), norm_cfg=norm_cfg)
            ]
            curr_numC = num_channels[i]
            layer.extend([BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg) for _ in range(num_layer[i] - 1)])
            features.append(nn.Sequential(*layer))
        self.BEV_features = nn.Sequential(*features)
        self.out_conv = _out_conv(num_channels[-1])  # PDBEV-17
        self.with_cp = with_cp

    def forward(self, x):
        """x: BEV features (B, C, Y, X) -> IFV (B, C', height_num, Y, X)."""
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.BEV_features):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        feats = self.out_conv(feats[-1])
        bev_height = feats[:, 0:self.height_num, ...].sigmoid()  # PDBEV-12: paper eq. 3
        bev_feature = feats[:, self.height_num:, ...]
        return bev_height.unsqueeze(1) * bev_feature.unsqueeze(2)
