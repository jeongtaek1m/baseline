# PCCR port: new file. BaseLSSFPN on our canonical cylindrical camera (port of CoIn3D/BEVDet/mmdet3d/canon/view_transformer.py).
"""Differences from BaseLSSFPN (everything else - image backbone / neck, DepthNet, depth bins d_bound, voxel pooling -
is the official code):

* frustum / get_geometry: the feature cell (i, j) samples the network-image pixel (u, v) of the official frustum grid
  (linspace(0, W_in - 1, W_feat), linspace(0, H_in - 1, H_feat), as create_frustum); the image-space augmentation is
  undone with ida_mats (only the flip and the vertical shift are used), giving the canonical pixel (u0, v0); its ray in
  the canonical camera frame is (sin a, t, cos a), a = (u0 - cx) / rho, t = (v0 - cy) / rho, and the depth bin d is the
  HORIZONTAL range along it: p_cam = d (sin a, t, cos a); p_ego = sensor2ego p_cam; then BDA. Computed in float32
  outside autocast (the official get_geometry runs under amp).
* the input size is read from the feature map, so the canonical canvas may differ between rigs (strip width); the
  frustum is built per call.
* cells whose frustum pixel is invalid in mats_dict['canon_valid'] (outside the source camera, or padding) are not
  lifted: their points are put at OUTSIDE, which the voxel-pooling kernels drop (bounds check).
* camera-aware DepthNet (mlp_input): the official 27 entries from the canonical parameters: intrinsics (rho, rho, 0, 0)
  - constant for every camera of every rig, as our BEVDet port (the principal point only says where the canvas starts;
  mlp_principal_point=True feeds it, ablation); sensor2ego is the levelled canonical pose (pitch / roll entries 0 / -1
  on every rig); ida carries the flip / shift; BDA unchanged.
"""
import torch
from torch.cuda.amp import autocast

from bevdepth.layers.backbones.base_lss_fpn import BaseLSSFPN

try:
    from bevdepth.ops.voxel_pooling_inference import voxel_pooling_inference
    from bevdepth.ops.voxel_pooling_train import voxel_pooling_train
except ImportError:
    print('Import VoxelPooling fail.')

__all__ = ['CanonLSSFPN', 'canon_rays', 'canon_points', 'apply_bda', 'OUTSIDE']

OUTSIDE = -1e4  # coordinate given to the points of invalid cells (outside the BEV grid)


def canon_rays(uv, sensor2ego, intrins, ida):
    """Network-image pixels -> canonical rays in the (key) ego frame (reference canon_rays, Megvii's ida form).

    uv (B, N, M, 2) pixels of the network image (after flip / shift / padding); sensor2ego (B, N, 4, 4) canonical
    camera -> ego; intrins (B, N, 3|4, 3|4) with rho = [0, 0] (= [1, 1]), cx = [0, 2], cy = [1, 2]; ida (B, N, 4, 4)
    with the 2 x 2 image-space map in [:2, :2] and its translation in [:2, 3].
    Returns rays (B, N, M, 3) in ego with unit horizontal norm (point at horizontal range r = origin + r * ray) and
    origin (B, N, 3).
    """
    A = ida[..., :2, :2]
    det = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
    A_inv = torch.stack([torch.stack([A[..., 1, 1], -A[..., 0, 1]], -1),
                         torch.stack([-A[..., 1, 0], A[..., 0, 0]], -1)], -2) / det[..., None, None]
    uv0 = torch.einsum('bnij,bnmj->bnmi', A_inv, uv - ida[..., None, :2, 3])
    rho_u, rho_v = intrins[..., 0, 0, None], intrins[..., 1, 1, None]
    cx, cy = intrins[..., 0, 2, None], intrins[..., 1, 2, None]
    a = (uv0[..., 0] - cx) / rho_u
    t = (uv0[..., 1] - cy) / rho_v
    ray_cam = torch.stack([torch.sin(a), t, torch.cos(a)], -1)
    rays = torch.einsum('bnij,bnmj->bnmi', sensor2ego[..., :3, :3], ray_cam)
    return rays, sensor2ego[..., :3, 3]


def apply_bda(points, bda):
    """points (B, ..., 3), bda (B, 4, 4) or None -> BDA-transformed points."""
    if bda is None:
        return points
    B = points.shape[0]
    shape = points.shape
    p = points.reshape(B, -1, 3)
    p = p @ bda[:, :3, :3].transpose(1, 2) + bda[:, None, :3, 3]
    return p.reshape(shape)


def canon_points(uv, r, sensor2ego, intrins, ida, bda):
    """Back-projection of pixels uv (B, N, M, 2) at known horizontal range r (B, N, M) -> ego points after BDA."""
    rays, origin = canon_rays(uv, sensor2ego, intrins, ida)
    return apply_bda(origin[:, :, None, :] + r[..., None] * rays, bda)


class CanonLSSFPN(BaseLSSFPN):

    def __init__(self, frustum_px='linspace', mlp_principal_point=False, **kwargs):
        super().__init__(**kwargs)
        assert frustum_px in ('linspace', 'center')
        self.frustum_px = frustum_px
        self.mlp_principal_point = mlp_principal_point
        self.register_buffer('depth_bins', self.frustum[:, 0, 0, 2].clone(), persistent=False)

    def pixel_grid(self, Hf, Wf, device, dtype):
        Hp, Wp = Hf * self.downsample_factor, Wf * self.downsample_factor
        if self.frustum_px == 'linspace':  # the official create_frustum grid
            u = torch.linspace(0, Wp - 1, Wf, device=device, dtype=dtype)
            v = torch.linspace(0, Hp - 1, Hf, device=device, dtype=dtype)
        else:
            u = (torch.arange(Wf, device=device, dtype=dtype) + 0.5) * self.downsample_factor - 0.5
            v = (torch.arange(Hf, device=device, dtype=dtype) + 0.5) * self.downsample_factor - 0.5
        return u, v

    @autocast(False)
    def get_canon_geometry(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat, feat_hw, valid=None):
        """(B, N, D, H_feat, W_feat, 3) ego coordinates (after BDA) of the canonical frustum points (replaces
        get_geometry); points of invalid cells are put at OUTSIDE."""
        sensor2ego_mat, intrin_mat, ida_mat = sensor2ego_mat.float(), intrin_mat.float(), ida_mat.float()
        bda_mat = bda_mat.float() if bda_mat is not None else None
        B, N = sensor2ego_mat.shape[:2]
        Hf, Wf = feat_hw
        u, v = self.pixel_grid(Hf, Wf, sensor2ego_mat.device, torch.float32)
        vv, uu = torch.meshgrid(v, u, indexing='ij')
        uv = torch.stack([uu, vv], -1).view(1, 1, Hf * Wf, 2).expand(B, N, Hf * Wf, 2)
        rays, origin = canon_rays(uv, sensor2ego_mat, intrin_mat, ida_mat)
        d = self.depth_bins.float().view(1, 1, -1, 1, 1)
        pts = origin[:, :, None, None, :] + d * rays[:, :, None, :, :]          # (B, N, D, M, 3)
        pts = apply_bda(pts, bda_mat).view(B, N, -1, Hf, Wf, 3)
        if valid is not None:
            Hp, Wp = Hf * self.downsample_factor, Wf * self.downsample_factor
            assert tuple(valid.shape[-2:]) == (Hp, Wp), f'valid mask {tuple(valid.shape)} vs input {(Hp, Wp)}'
            vi = v.round().long().clamp(0, Hp - 1)
            ui = u.round().long().clamp(0, Wp - 1)
            cell_ok = valid.view(B, N, Hp, Wp)[:, :, vi][:, :, :, ui] > 0             # (B, N, Hf, Wf)
            pts = torch.where(cell_ok[:, :, None, :, :, None], pts, pts.new_full((), OUTSIDE))
        return pts

    def _forward_depth_net(self, feat, mats_dict):
        """Official DepthNet; its camera-aware MLP gets the canonical intrinsics (rho, rho, 0, 0)."""
        if not self.mlp_principal_point:
            intrin = mats_dict['intrin_mats'].clone()
            intrin[..., 0, 2] = 0.0
            intrin[..., 1, 2] = 0.0
            mats_dict = dict(mats_dict, intrin_mats=intrin)
        return self.depth_net(feat, mats_dict)

    def _forward_single_sweep(self, sweep_index, sweep_imgs, mats_dict, is_return_depth=False):
        """BaseLSSFPN._forward_single_sweep with the canonical geometry (official code otherwise)."""
        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape
        img_feats = self.get_cam_feats(sweep_imgs)
        source_features = img_feats[:, 0, ...]
        depth_feature = self._forward_depth_net(
            source_features.reshape(batch_size * num_cams,
                                    source_features.shape[2],
                                    source_features.shape[3],
                                    source_features.shape[4]),
            mats_dict,
        )
        depth = depth_feature[:, :self.depth_channels].softmax(
            dim=1, dtype=depth_feature.dtype)
        geom_xyz = self.get_canon_geometry(  # PCCR port: canonical frustum from the feature map size
            mats_dict['sensor2ego_mats'][:, sweep_index, ...],
            mats_dict['intrin_mats'][:, sweep_index, ...],
            mats_dict['ida_mats'][:, sweep_index, ...],
            mats_dict.get('bda_mat', None),
            tuple(source_features.shape[-2:]),
            mats_dict.get('canon_valid', None),
        )
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) /
                    self.voxel_size).int()
        if self.training or self.use_da:
            img_feat_with_depth = depth.unsqueeze(
                1) * depth_feature[:, self.depth_channels:(
                    self.depth_channels + self.output_channels)].unsqueeze(2)

            img_feat_with_depth = self._forward_voxel_net(img_feat_with_depth)

            img_feat_with_depth = img_feat_with_depth.reshape(
                batch_size,
                num_cams,
                img_feat_with_depth.shape[1],
                img_feat_with_depth.shape[2],
                img_feat_with_depth.shape[3],
                img_feat_with_depth.shape[4],
            )

            img_feat_with_depth = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2)

            feature_map = voxel_pooling_train(geom_xyz,
                                              img_feat_with_depth.contiguous(),
                                              self.voxel_num.cuda())
        else:
            feature_map = voxel_pooling_inference(
                geom_xyz, depth, depth_feature[:, self.depth_channels:(
                    self.depth_channels + self.output_channels)].contiguous(),
                self.voxel_num.cuda())
        if is_return_depth:
            # final_depth has to be fp32, otherwise the depth
            # loss will colapse during the traing process.
            return feature_map.contiguous(
            ), depth_feature[:, :self.depth_channels].softmax(dim=1)
        return feature_map.contiguous()
