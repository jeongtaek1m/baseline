# UDGA port (Gen2E): new file. BaseLSSFPN with UDGA's (DG-BEV) intrinsic-normalised depth bins.
# UDGA is Samsung-proprietary; this is a re-implementation and stays in the private repository.
"""Port of the old-base LSSViewTransformerBEVDepthUDGA (bevdepth/mmdet3d/models/necks/view_transformer_udga.py) to the
Megvii LSS backbone. UDGA's config lifts with DG-BEV's virtual depth (pixel_const 400): the depth bins (d_bound,
UDGA: [1, 90, 1]) are virtual depths v and the metric depth is

    d = v / k,   k = pixel_const * sqrt(fx'^-2 + fy'^-2),

with fx', fy' the diagonal of ida[:3, :3] @ K (BEVDet: post_rots @ intrins; DG-BEV eq. 3 with c = 1 / pixel_const).
Everything else is BaseLSSFPN's (R50 + SECONDFPN, the 27-dim camera-aware DepthNet with DCN + ASPP, voxel pooling).
forward(..., return_logits=True) also returns the depth logits, because UDGA's overlap loss compares logits; the lift
still uses the softmax.
"""
import torch

from bevdepth.layers.backbones.base_lss_fpn import BaseLSSFPN

try:
    from bevdepth.ops.voxel_pooling_inference import voxel_pooling_inference
    from bevdepth.ops.voxel_pooling_train import voxel_pooling_train
except ImportError:
    print('Import VoxelPooling fail.')

__all__ = ['UDGALSSFPN']


class UDGALSSFPN(BaseLSSFPN):

    def __init__(self, pixel_const=400.0, **kwargs):
        super().__init__(**kwargs)
        assert not self.use_da
        self.pixel_const = pixel_const

    def depth_scale(self, intrin_mat, ida_mat):
        """k of every camera, (B, N): metric depth = virtual depth / k."""
        new_intrins = torch.matmul(ida_mat[..., :3, :3], intrin_mat[..., :3, :3])
        return self.pixel_const * torch.sqrt(1.0 / new_intrins[..., 0, 0]**2 + 1.0 / new_intrins[..., 1, 1]**2)

    def get_geometry(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat):
        """BaseLSSFPN.get_geometry with the frustum's virtual depth divided by k: (B, N, D, H, W, 3)."""
        batch_size, num_cams, _, _ = sensor2ego_mat.shape
        points = self.frustum
        ida_view = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = ida_view.inverse().matmul(points.unsqueeze(-1))
        # virtual -> metric depth (the ida matrix leaves the depth row unchanged)
        k = self.depth_scale(intrin_mat, ida_mat).view(batch_size, num_cams, 1, 1, 1, 1, 1)
        points = torch.cat((points[..., :2, :], points[..., 2:3, :] / k, points[..., 3:, :]), 5)
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

    def forward(self, sweep_imgs, mats_dict, timestamps=None, is_return_depth=False, return_logits=False):
        """Single key frame (BaseLSSFPN._forward_single_sweep with the UDGA geometry)."""
        batch_size, num_sweeps, num_cams, num_channels, img_height, img_width = sweep_imgs.shape
        assert num_sweeps == 1
        img_feats = self.get_cam_feats(sweep_imgs)
        source_features = img_feats[:, 0, ...]
        depth_feature = self._forward_depth_net(
            source_features.reshape(batch_size * num_cams, source_features.shape[2], source_features.shape[3],
                                    source_features.shape[4]), mats_dict)
        depth = depth_feature[:, :self.depth_channels].softmax(dim=1, dtype=depth_feature.dtype)
        geom_xyz = self.get_geometry(mats_dict['sensor2ego_mats'][:, 0], mats_dict['intrin_mats'][:, 0],
                                     mats_dict['ida_mats'][:, 0], mats_dict.get('bda_mat', None))
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) / self.voxel_size).int()
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
        if return_logits:
            return feature_map, depth_feature[:, :self.depth_channels]
        if is_return_depth:
            return feature_map, depth_feature[:, :self.depth_channels].softmax(dim=1)
        return feature_map
