# PD-BEV port (Gen2E): new file. BaseLSSFPN with PD-BEV's virtual depth and camera input.
"""The Megvii LSS backbone (R50 + SECONDFPN, camera-aware DepthNet with DCN + ASPP, voxel pooling) with the two
changes PD-BEV makes to its view transformer (official LSSViewTransformer_pc, bevdepth/mmdet3d/models/necks/
view_transformer_pc.py):

- virtual depth: the depth bins (d_bound, PD-BEV: [1, 100, 1]) are virtual depths; a frustum point's metric depth is
  d_virtual / k with k = sqrt(100 / fx'^2 + 100 / fy'^2) / sqrt(100 / 450^2 + 100 / 450^2) from the actual
  (augmented) focal lengths (official get_lidar_coor);
- the camera-aware input of the DepthNet has 31 values: the official 27 (fx, fy, cx, cy, ida, bda, sensor2ego) plus
  the actual intrinsics fx', fy', cx', cy' (official get_mlp_input; order as official).

Everything else is BaseLSSFPN's. forward(..., return_pdbev=True) also returns what the PD-BEV losses need: the image
features (Img_Aux), the camera input (Img_Aux_Dy) and the frustum points in the (bda-augmented) ego frame (rendering).
"""
import torch
from torch import nn

from bevdepth.layers.backbones.base_lss_fpn import BaseLSSFPN, DepthNet, Mlp

try:
    from bevdepth.ops.voxel_pooling_inference import voxel_pooling_inference
    from bevdepth.ops.voxel_pooling_train import voxel_pooling_train
except ImportError:
    print('Import VoxelPooling fail.')

__all__ = ['PDBEVLSSFPN', 'PDBEVDepthNet']

F_REF = 450.0


class PDBEVDepthNet(DepthNet):
    """Megvii's DepthNet (DCN + ASPP) with PD-BEV's 31-dim camera input; the input is built by the backbone."""

    def __init__(self, in_channels, mid_channels, context_channels, depth_channels, mlp_channels=31):
        super().__init__(in_channels, mid_channels, context_channels, depth_channels)
        self.bn = nn.BatchNorm1d(mlp_channels)
        self.depth_mlp = Mlp(mlp_channels, mid_channels, mid_channels)
        self.context_mlp = Mlp(mlp_channels, mid_channels, mid_channels)

    def forward(self, x, mlp_input):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)
        return torch.cat([depth, context], dim=1)


class PDBEVLSSFPN(BaseLSSFPN):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert not self.use_da
        self.c_ref = float((100.0 / F_REF**2 + 100.0 / F_REF**2)**0.5)

    def _configure_depth_net(self, depth_net_conf):
        return PDBEVDepthNet(depth_net_conf['in_channels'], depth_net_conf['mid_channels'], self.output_channels,
                             self.depth_channels)

    def get_mlp_input(self, mats_dict, sensor2ego=None):
        """Official get_mlp_input: (B, N, 31). sensor2ego (B, N, 4, 4) replaces the real cameras' (virtual views)."""
        intrin = mats_dict['intrin_mats'][:, 0, ..., :3, :3]
        actual = mats_dict['actual_intrin_mats'][:, 0, ..., :3, :3]
        ida = mats_dict['ida_mats'][:, 0]
        if sensor2ego is None:
            sensor2ego = mats_dict['sensor2ego_mats'][:, 0]
        B, N = intrin.shape[:2]
        bda = mats_dict['bda_mat'].view(B, 1, 4, 4).repeat(1, N, 1, 1)
        mlp_input = torch.stack([
            intrin[..., 0, 0], intrin[..., 1, 1], intrin[..., 0, 2], intrin[..., 1, 2],
            actual[..., 0, 0], actual[..., 1, 1], actual[..., 0, 2], actual[..., 1, 2],
            ida[..., 0, 0], ida[..., 0, 1], ida[..., 0, 3], ida[..., 1, 0], ida[..., 1, 1], ida[..., 1, 3],
            bda[..., 0, 0], bda[..., 0, 1], bda[..., 1, 0], bda[..., 1, 1], bda[..., 2, 2],
        ], dim=-1)
        return torch.cat([mlp_input, sensor2ego[..., :3, :].reshape(B, N, 12)], dim=-1)

    def get_geometry(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat, actual_intrin_mat=None):
        """BaseLSSFPN.get_geometry with PD-BEV's virtual -> metric depth (official get_lidar_coor): (B, N, D, H, W, 3)."""
        assert actual_intrin_mat is not None, 'PD-BEV needs the actual intrinsics'
        batch_size, num_cams, _, _ = sensor2ego_mat.shape
        points = self.frustum
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = ida_mat.inverse().matmul(points.unsqueeze(-1))
        k = torch.sqrt(100.0 / torch.square(actual_intrin_mat[..., 0, 0]) +
                       100.0 / torch.square(actual_intrin_mat[..., 1, 1])) / self.c_ref
        points = torch.cat((points[..., :2, :], points[..., 2:3, :] / k.view(batch_size, num_cams, 1, 1, 1, 1, 1),
                            points[..., 3:, :]), 5)
        # cam_to_ego
        points = torch.cat((points[..., :2, :] * points[..., 2:3, :], points[..., 2:, :]), 5)
        combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
        points = combine.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points)
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)
        return points[..., :3]

    def forward(self, sweep_imgs, mats_dict, timestamps=None, is_return_depth=False, return_pdbev=False):
        """Single key frame (BaseLSSFPN._forward_single_sweep with the PD-BEV geometry and camera input)."""
        batch_size, num_sweeps, num_cams, num_channels, img_height, img_width = sweep_imgs.shape
        assert num_sweeps == 1
        img_feats = self.get_cam_feats(sweep_imgs)[:, 0, ...]  # (B, N, C, h, w)
        mlp_input = self.get_mlp_input(mats_dict)
        depth_feature = self.depth_net(
            img_feats.reshape(batch_size * num_cams, img_feats.shape[2], img_feats.shape[3], img_feats.shape[4]),
            mlp_input)
        depth = depth_feature[:, :self.depth_channels].softmax(dim=1, dtype=depth_feature.dtype)
        geom = self.get_geometry(mats_dict['sensor2ego_mats'][:, 0], mats_dict['intrin_mats'][:, 0],
                                 mats_dict['ida_mats'][:, 0], mats_dict.get('bda_mat', None),
                                 mats_dict['actual_intrin_mats'][:, 0])
        geom_xyz = ((geom - (self.voxel_coord - self.voxel_size / 2.0)) / self.voxel_size).int()
        context = depth_feature[:, self.depth_channels:(self.depth_channels + self.output_channels)]
        if self.training:
            img_feat_with_depth = depth.unsqueeze(1) * context.unsqueeze(2)
            img_feat_with_depth = img_feat_with_depth.reshape(batch_size, num_cams, img_feat_with_depth.shape[1],
                                                              img_feat_with_depth.shape[2],
                                                              img_feat_with_depth.shape[3],
                                                              img_feat_with_depth.shape[4])
            img_feat_with_depth = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2)
            feature_map = voxel_pooling_train(geom_xyz, img_feat_with_depth.contiguous(), self.voxel_num.cuda())
        else:
            feature_map = voxel_pooling_inference(geom_xyz, depth, context.contiguous(), self.voxel_num.cuda())
        feature_map = feature_map.contiguous()
        if return_pdbev:
            # the depth for the loss has to be fp32 (official comment in BaseLSSFPN)
            return dict(bev=feature_map, depth=depth_feature[:, :self.depth_channels].softmax(dim=1),
                        img_feats=img_feats, mlp_input=mlp_input, geom=geom)
        if is_return_depth:
            return feature_map, depth_feature[:, :self.depth_channels].softmax(dim=1)
        return feature_map
