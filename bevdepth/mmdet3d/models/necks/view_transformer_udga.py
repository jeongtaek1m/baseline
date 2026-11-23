# Gen2E: UDGA (NeurIPS 2024) re-implementation on the vanilla BEVDepth path.
# The UDGA code is Samsung-proprietary; this file is a re-implementation and stays in the
# private repository (README, license notes).
import torch

from ..builder import NECKS
from .view_transformer import LSSViewTransformerBEVDepth


@NECKS.register_module()
class LSSViewTransformerBEVDepthUDGA(LSSViewTransformerBEVDepth):
    """Vanilla BEVDepth view transformer with UDGA's intrinsic-normalised depth bins.

    UDGA's config (bevssl3sup_lyf_to_nus_cls1_d60_pc400_L1_extaug.py) lifts with DG-BEV's
    virtual depth (`pixel_const=400`, ViewTransformerLiftSplatShoot_SSL.get_geometry):
    the depth bins `grid_config['depth']` are virtual depths v and the metric depth is

        d = v / k,   k = pixel_const * sqrt(fx'^-2 + fy'^-2),

    with fx', fy' the diagonal of post_rots @ intrins (DG-BEV eq. 3 with c = 1 / pixel_const).
    PCCR R1 at the 704-px input has k = 1.02.

    Everything else is the vanilla LSSViewTransformerBEVDepth (DepthNet, BEVPoolv2).
    `forward` returns the depth logits instead of the softmax, because UDGA's overlap loss
    compares logits (config `ssl_loss='l1_loss'`); the lift still uses the softmax.
    The main Gen2E UDGA run has no LiDAR depth (decision 2026-11-23); the UDGA-Ld variant
    uses get_depth_loss_virtual.
    """

    def __init__(self, pixel_const=400.0, **kwargs):
        super().__init__(**kwargs)
        assert not self.accelerate, 'per-sample depth scaling needs accelerate=False'
        self.pixel_const = pixel_const

    def depth_scale(self, cam2imgs, post_rots):
        """k of every camera, (B, N): metric depth = virtual depth / k."""
        new_intrins = torch.matmul(post_rots, cam2imgs)
        return self.pixel_const * torch.sqrt(
            1.0 / new_intrins[..., 0, 0] ** 2 + 1.0 / new_intrins[..., 1, 1] ** 2)

    def get_lidar_coor(self, rots, trans, cam2imgs, post_rots, post_trans, bda):
        """LSSViewTransformer.get_lidar_coor with the frustum depth divided by k."""
        B, N, _ = trans.shape
        points = self.frustum.to(rots) - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1))
        # virtual -> metric depth (post_rots/post_trans leave the depth row unchanged)
        k = self.depth_scale(cam2imgs, post_rots).view(B, N, 1, 1, 1, 1, 1)
        points = torch.cat((points[..., :2, :], points[..., 2:3, :] / k), 5)
        # cam_to_ego
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = rots.matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        points = bda.view(B, 1, 1, 1, 1, 3,
                          3).matmul(points.unsqueeze(-1)).squeeze(-1)
        return points

    def get_depth_loss_virtual(self, gt_depths, depth_preds, cam2imgs, post_rots):
        """Vanilla BEVDepth depth loss with the LiDAR depth moved into the virtual-depth bins.

        UDGA-2: the official target is d / k while its lift is v / k. Here the target is
        v = d * k (DG-BEV eq. 3), exactly what get_lidar_coor inverts. gt_depths: metric
        LiDAR depth maps (B, N, H, W) from PointToMultiViewDepth (0 = no point); k is constant
        per camera, so it commutes with the min-pooling in get_downsampled_gt_depth.
        depth_preds: softmax probabilities (B*N, D, fH, fW), as the vanilla loss expects.
        """
        k = self.depth_scale(cam2imgs, post_rots)
        return self.get_depth_loss(gt_depths * k[..., None, None], depth_preds)

    def forward(self, input):
        (x, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input) = input[:8]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x, mlp_input)
        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1)
        bev_feat, _ = self.view_transform(input, depth, tran_feat)
        return bev_feat, depth_digit
