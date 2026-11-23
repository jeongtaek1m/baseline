# Gen2E: UDGA (NeurIPS 2024) re-implementation on the vanilla BEVDepth path.
# The UDGA code is Samsung-proprietary; this file is a re-implementation and stays in the
# private repository (README, license notes). References are to third_party/UDGA (@9198218).
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import DETECTORS
from .bevdet import BEVDepth4D


@torch.no_grad()
def frustum_correspondences(view_transformer, rots, trans, intrins, post_rots, post_trans):
    """Corresponding frustum cells between all ordered camera pairs of each sample.

    Re-implements the geometry of UDGA's overlap constraint (BEVSSL3.reprojection,
    bevssl.py:51-129) with the bugs fixed (docs/bug_ledger.md):
      UDGA-1  the 3D point of a cell and the logit attached to it come from the same
              (d, h, w) cell: the points are the view transformer's own frustum
              (D, fH, fW), in the logits' memory order.
      UDGA-3  a cell's point is the lift's point: pixel grid of the frustum (linspace),
              metric depth = virtual depth / k (the same get_lidar_coor as the lift).
      UDGA-7  no hard-coded pairs: every ordered pair (i, j), i != j, with at least one
              cell of camera i whose point falls into camera j's frustum.
      UDGA-8  B and N come from the tensors.
    A point of camera i corresponds to the cell of camera j whose pixel is nearest
    (frustum grid) and whose virtual-depth interval [d0 + b*step, d0 + (b+1)*step) holds
    the point's virtual depth in camera j (the interval convention of BEVDepth's depth
    labels). No gradient flows through the geometry.

    Returns a dict with flat indices into logits.reshape(-1), logits of shape
    (B*N, D, fH, fW):
      src   (P,) source cells, one per valid point
      dst   (P,) the corresponding target cell of each point
      key   (P,) one id per (sample, pair i->j, target cell); points sharing a key are
            averaged before the comparison (UDGA-6)
      pairs list over samples of the ordered pairs (i, j) with at least one point
    """
    vt = view_transformer
    B, N = trans.shape[:2]
    D, fH, fW, _ = vt.frustum.shape
    d0, _, step = vt.grid_config['depth']
    eye = torch.eye(3, device=rots.device, dtype=rots.dtype).expand(B, 3, 3)
    pts = vt.get_lidar_coor(rots, trans, intrins, post_rots, post_trans, eye)  # (B,N,D,fH,fW,3) ego
    k = vt.depth_scale(intrins, post_rots)  # (B, N)
    # image -> feature grid: the frustum samples pixel x = c * (W_in - 1) / (fW - 1)
    w_in1 = float(vt.frustum[0, 0, -1, 0])
    h_in1 = float(vt.frustum[0, -1, 0, 1])
    cells = N * D * fH * fW
    src_cam = torch.arange(N, device=rots.device).view(N, 1, 1, 1).expand(N, D, fH, fW)
    local = torch.arange(cells, device=rots.device).view(N, D, fH, fW)
    src_l, dst_l, key_l, pairs = [], [], [], []
    for b in range(B):
        p = pts[b].reshape(-1, 3)
        pairs_b = []
        for j in range(N):
            cam = (p - trans[b, j]) @ rots[b, j]  # ego -> camera j (rots: cam -> ego)
            z = cam[:, 2]
            uv = cam @ intrins[b, j].T
            uvz = torch.stack([uv[:, 0] / z.clamp(min=1e-6), uv[:, 1] / z.clamp(min=1e-6), z], 1)
            uvz = uvz @ post_rots[b, j].T + post_trans[b, j]
            col = torch.round(uvz[:, 0] * (fW - 1) / w_in1).long()
            row = torch.round(uvz[:, 1] * (fH - 1) / h_in1).long()
            dbin = torch.floor((z * k[b, j] - d0) / step).long()
            ok = ((z > 1e-3) & (col >= 0) & (col < fW) & (row >= 0) & (row < fH)
                  & (dbin >= 0) & (dbin < D) & (src_cam.reshape(-1) != j))
            if not bool(ok.any()):
                continue
            src = local.reshape(-1)[ok]
            dst = ((j * D + dbin[ok]) * fH + row[ok]) * fW + col[ok]
            i = src_cam.reshape(-1)[ok]
            pair = i * N + j
            src_l.append(src + b * cells)
            dst_l.append(dst + b * cells)
            key_l.append((b * N * N + pair) * cells + dst)
            pairs_b.extend((int(a), j) for a in torch.unique(i).tolist())
        pairs.append(sorted(pairs_b))
    if src_l:
        src, dst, key = torch.cat(src_l), torch.cat(dst_l), torch.cat(key_l)
    else:
        src = dst = key = torch.zeros(0, dtype=torch.long, device=rots.device)
    return dict(src=src, dst=dst, key=key, pairs=pairs)


def overlap_loss(depth_logits, corr, beta=1.0):
    """UDGA's multi-view overlap depth constraint L_ov (paper eq. 2-3), corresponding cells only.

    For each pair i->j the logits of camera i are moved to camera j's frustum cells, the
    values landing in the same cell are averaged (UDGA-6: the official scatter is
    last-write-wins), and SmoothL1 compares them with camera j's logits in those cells only
    (UDGA-9: the official loss also compares every other depth bin of the pixel against 0).
    Logits, SmoothL1 (beta 1) and no detach follow the official config (`ssl_loss='l1_loss'`).
    An empty set gives 0 instead of NaN (UDGA-6).
    """
    flat = depth_logits.reshape(-1)
    if corr['key'].numel() == 0:
        return flat.sum() * 0.0
    uniq, inv = torch.unique(corr['key'], return_inverse=True)
    ones = torch.ones_like(corr['src'], dtype=flat.dtype)
    cnt = flat.new_zeros(len(uniq)).index_add_(0, inv, ones)
    sums = flat.new_zeros(len(uniq)).index_add_(0, inv, flat[corr['src']])
    dst = torch.zeros_like(uniq).scatter_(0, inv, corr['dst'])
    return F.smooth_l1_loss(sums / cnt, flat[dst], beta=beta)


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images (UDGA bevssl.py:627-658)."""

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


@torch.no_grad()
def photo_reprojection_public(depth_pred, img, rots, trans, intrins, post_rots, post_trans,
                              pairs, dbound, downsample=16):
    """UDGA BEVSSL3.photo_reprojection (bevssl.py:170-223), kept as the public code.

    User decision 2026-11-23 ("일단 공개 코드 대로"): L_p stays as released, so it has no
    gradient (argmax depth, integer indexing; UDGA-4) and also keeps the released code's
    other quirks: the argmax bin index is used as metric depth (no pixel_const, off by
    dbound[0]) and the (w, h)-ordered pixel grid is paired with the (h, w)-ordered depth
    (the UDGA-1 transposition). Only what PCCR needs to run changed: `pairs` replaces the
    hard-coded 6-camera table cam_pair_idx (UDGA-7), and B, N come from the tensors (UDGA-8).
    """
    device = depth_pred.device
    B, N, _, H, W = img.shape
    BN, D, height, width = depth_pred.shape
    depth = torch.argmax(depth_pred.sigmoid(), dim=1)

    coor_w = torch.linspace(0, width - 1, width)
    coor_h = torch.linspace(0, height - 1, height)
    coor_w, coor_h = torch.meshgrid(coor_w, coor_h, indexing='ij')
    wh = torch.stack([coor_w * downsample,
                      coor_h * downsample], dim=-1).expand(B, N, width, height, 2).reshape(
                          B, N, width * height, 2)
    whd = torch.cat([wh.to(device), depth.view(B, N, width * height, 1).to(wh.dtype)], dim=-1)

    src_img, tar_img = [], []
    for b in range(B):
        for cam, pid in pairs[b]:
            img_org = whd[b, cam]
            img_s = img[b, cam].permute(1, 2, 0)
            img_t = img[b, pid].permute(1, 2, 0)
            # Depth to Point
            depth_img = (img_org - post_trans[b, cam]).matmul(torch.inverse(post_rots[b, cam]).T)
            depth_pts = torch.cat([depth_img[:, :2] * depth_img[:, 2:3], depth_img[:, 2:3]], 1)
            combine = rots[b, cam].matmul(torch.inverse(intrins[b, cam]))
            depth_pts = depth_pts.matmul(combine.T) + trans[b, cam]

            # Point to adjacent cam Depth
            combine_inv = torch.inverse(rots[b, pid].matmul(torch.inverse(intrins[b, pid])))
            adj_depth = (depth_pts[:, :3] - trans[b, pid:pid + 1, :]).matmul(combine_inv.T)
            adj_depth = torch.cat([adj_depth[:, :2] / adj_depth[:, 2:3],
                                   adj_depth[:, 2:3]], 1)
            adj_depth = adj_depth.matmul(post_rots[b, pid].T) + post_trans[b, pid:pid + 1, :]
            adj_depth[:, :2] = torch.round(adj_depth[:, :2])

            kept1 = (adj_depth[:, 0] >= 0) & (adj_depth[:, 0] < W) \
                & (adj_depth[:, 1] >= 0) & (adj_depth[:, 1] < H) \
                & (adj_depth[:, 2] < dbound[1]) \
                & (adj_depth[:, 2] >= dbound[0])

            coor_s = torch.round(img_org[kept1]).to(torch.long)
            img_s = img_s[coor_s[:, 1], coor_s[:, 0], :]
            coor_t = torch.round(adj_depth[kept1]).to(torch.long)
            img_t = img_t[coor_t[:, 1], coor_t[:, 0], :]
            src_img.append(img_s)
            tar_img.append(img_t)
    if not src_img:
        return None, None
    src_img = torch.cat(src_img, dim=0)
    tar_img = torch.cat(tar_img, dim=0)
    return src_img.view(1, 1, -1, 3), tar_img.view(1, 1, -1, 3)


# mmlabNormalize (loading_upstream.py): RGB, (x - mean) / std on 0..255 values
IMG_MEAN = (123.675, 116.28, 103.53)
IMG_STD = (58.395, 57.12, 57.375)


def expected_metric_depth(depth_logits, view_transformer, intrins, post_rots, out_hw):
    """Expected metric depth of every image pixel, (B, N, H, W), differentiable.

    E[v] over the softmax of the virtual-depth bins (each bin at its frustum depth
    d0 + b * step, as the lift places it), divided by k (DG-BEV eq. 3), then bilinearly
    upsampled from the feature grid to the image with align_corners=True, which matches
    the frustum's pixel grid linspace(0, W_in - 1, fW).
    """
    vt = view_transformer
    B, N = intrins.shape[:2]
    BN, D, fh, fw = depth_logits.shape
    d0, _, step = vt.grid_config['depth']
    bins = d0 + step * torch.arange(D, device=depth_logits.device, dtype=depth_logits.dtype)
    ev = (depth_logits.softmax(1) * bins.view(1, D, 1, 1)).sum(1, keepdim=True)  # (BN,1,fh,fw)
    em = ev / vt.depth_scale(intrins, post_rots).reshape(BN, 1, 1, 1)
    return F.interpolate(em, size=out_hw, mode='bilinear', align_corners=True).view(B, N, *out_hw)


def photometric_error_maps(depth, rgb, rots, trans, intrins, post_rots, post_trans, pairs, ssim):
    """Per-pair photometric error of UDGA's L_p (paper eq. 4), differentiable in `depth`.

    depth (B, N, H, W): metric depth of camera i's pixels (augmented image grid).
    rgb (B, N, 3, H, W): images in [0, 1]. For each ordered pair (i, j) of `pairs[b]`, every
    pixel p of camera i is back-projected with depth(p) (inverse image augmentation, K_i^-1,
    sensor2ego of i), projected into camera j (inverse sensor2ego of j, K_j, augmentation of j)
    and I_j is bilinearly sampled there (eq. 4: <.> = bilinear sampling). The error is the public
    code's mix, pe = 0.85 * (1 - SSIM) / 2 + 0.15 * |I_i - I_j<p*>| (bevssl.py:225-233), with
    the public 3x3 SSIM on the full images. A pixel counts when its projection lies inside
    image j in front of the camera, and neither image is augmentation padding (black) there.
    Returns a list over (b, i, j) of (pe (H, W), valid (H, W) bool).
    """
    B, N, _, H, W = rgb.shape
    dev, dt = depth.device, depth.dtype
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=dt),
                            torch.arange(W, device=dev, dtype=dt), indexing='ij')
    pix = torch.stack([xs, ys, torch.ones_like(xs)], -1).view(-1, 3)
    content = (rgb.sum(2, keepdim=True) > 1e-3).to(dt)  # (B, N, 1, H, W); padding is black
    out = []
    for b in range(B):
        for i, j in pairs[b]:
            p = (pix - post_trans[b, i]) @ torch.inverse(post_rots[b, i]).T  # un-augmented, z = 1
            d = depth[b, i].reshape(-1, 1)
            cam_i = torch.cat([p[:, :2] * d, d], 1) @ torch.inverse(intrins[b, i]).T
            cam_j = (cam_i @ rots[b, i].T + trans[b, i] - trans[b, j]) @ rots[b, j]
            z = cam_j[:, 2:3]
            uv = cam_j @ intrins[b, j].T
            uvz = torch.cat([uv[:, :2] / z.clamp(min=1e-3), z], 1) @ post_rots[b, j].T + post_trans[b, j]
            gx = uvz[:, 0] / (W - 1) * 2 - 1
            gy = uvz[:, 1] / (H - 1) * 2 - 1
            grid = torch.stack([gx, gy], -1).view(1, H, W, 2)
            warped = F.grid_sample(rgb[b, j:j + 1], grid, mode='bilinear', padding_mode='zeros',
                                   align_corners=True)
            warped_content = F.grid_sample(content[b, j:j + 1], grid, mode='bilinear',
                                           padding_mode='zeros', align_corners=True)
            target = rgb[b, i:i + 1]
            l1 = (warped - target).abs().mean(1)[0]
            ssim_map = ssim(warped, target).mean(1)[0]
            pe = 0.85 * ssim_map + 0.15 * l1
            valid = ((z.view(H, W) > 1e-3) & (gx.view(H, W).abs() <= 1) & (gy.view(H, W).abs() <= 1)
                     & (warped_content[0, 0] > 0.999) & (content[b, i, 0] > 0.5)).detach()
            out.append((pe, valid))
    return out


def photometric_loss_paper(depth_logits, imgs, rots, trans, intrins, post_rots, post_trans,
                           pairs, view_transformer, ssim, weight=2.0):
    """UDGA L_p as the paper writes it (eq. 4), so that it has a gradient (UDGA-4 fixed).

    Expected metric depth from the depth logits, the normalised network images turned back
    into [0, 1] RGB, photometric_error_maps over the overlapping pairs, mean over the valid
    pixels of all pairs in the batch, times `weight` (2.0, the public code's value). Returns
    a zero that keeps the graph when no pixel is valid.
    """
    H, W = imgs.shape[-2:]
    mean = imgs.new_tensor(IMG_MEAN).view(1, 1, 3, 1, 1)
    std = imgs.new_tensor(IMG_STD).view(1, 1, 3, 1, 1)
    rgb = ((imgs * std + mean) / 255.0).clamp(0, 1)
    depth = expected_metric_depth(depth_logits, view_transformer, intrins, post_rots, (H, W))
    maps = photometric_error_maps(depth, rgb, rots, trans, intrins, post_rots, post_trans, pairs, ssim)
    vals = [pe[v] for pe, v in maps if bool(v.any())]
    if not vals:
        return depth_logits.sum() * 0.0
    return weight * torch.cat(vals).mean()


@DETECTORS.register_module()
class BEVDepthUDGA(BEVDepth4D):
    """BEVDepth (single frame, as the vanilla config) + UDGA domain-generalisation losses.

    UDGA pre-training stage (paper sec. 3.3, 3.5; config bevssl3sup_..._pc400_L1_extaug.py),
    as decided for Gen2E on 2026-11-23 (docs/decisions.md):
      L = L_det + overlap_loss_weight * L_ov + L_p [+ L_d]
      * L_ov: overlap_loss() above, weight 1 (official ssl_weight)
      * L_p, photo_loss_mode:
          'public' the public code's term (photo_reprojection_public), weight 2.0; no
                   gradient, so it is logged but does not change training (main run)
          'paper'  photometric_loss_paper (eq. 4 with bilinear sampling), weight 2.0 (UDGA-Lp)
      * L_d, lidar_depth_loss: False = no LiDAR depth (paper table 4 / eq. 7; main run and
        UDGA-Lp); True = the vanilla BEVDepth depth loss on virtual-depth targets, as the
        public config has LiDAR depth on (UDGA-Ld; needs gt_depth in the pipeline)
      * virtual depth bins: LSSViewTransformerBEVDepthUDGA (pixel_const 400)
      * ext aug: LoadAnnotationsBEVDepthExtAug in the pipeline
    LEDA (adapters) is the label-efficient adaptation stage and is not part of DG.
    """

    def __init__(self, overlap_loss_weight=1.0, overlap_loss_beta=1.0,
                 photo_loss_weight=2.0, photo_loss_mode='public', lidar_depth_loss=False,
                 **kwargs):
        super().__init__(**kwargs)
        assert self.num_frame == 1, 'UDGA here is single-frame (num_adj=0)'
        assert photo_loss_mode in ('public', 'paper')
        self.overlap_loss_weight = overlap_loss_weight
        self.overlap_loss_beta = overlap_loss_beta
        self.photo_loss_weight = photo_loss_weight
        self.photo_loss_mode = photo_loss_mode
        self.lidar_depth_loss = lidar_depth_loss
        self.compute_ssim_loss = SSIM()

    def get_rgb_loss(self, rgb1, rgb2, weight=2.0):
        """UDGA BEVSSL3.get_rgb_loss (bevssl.py:225-233)."""
        diff_img = (rgb1 - rgb2).abs()
        diff_img = diff_img.clamp(0, 1)

        ssim_map = self.compute_ssim_loss(rgb1, rgb2)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)
        b, h, w, c = diff_img.shape
        loss = weight * diff_img.sum() / (b * h * w * c)
        return loss

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        img_feats, _, depth_logits = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        imgs, rots, trans, intrins, post_rots, post_trans, _ = \
            self.prepare_inputs(img_inputs)
        cam = (rots[0], trans[0], intrins[0], post_rots[0], post_trans[0])
        corr = frustum_correspondences(self.img_view_transformer, *cam)
        losses = dict(loss_overlap=self.overlap_loss_weight * overlap_loss(
            depth_logits, corr, beta=self.overlap_loss_beta))
        if self.lidar_depth_loss:
            losses['loss_depth'] = self.img_view_transformer.get_depth_loss_virtual(
                kwargs['gt_depth'], depth_logits.softmax(1), intrins[0], post_rots[0])
        if self.photo_loss_weight > 0 and self.photo_loss_mode == 'paper':
            losses['loss_rgb'] = photometric_loss_paper(
                depth_logits, imgs[0], *cam, pairs=corr['pairs'],
                view_transformer=self.img_view_transformer, ssim=self.compute_ssim_loss,
                weight=self.photo_loss_weight)
        elif self.photo_loss_weight > 0:
            rgb_s, rgb_t = photo_reprojection_public(
                depth_logits, imgs[0], *cam, pairs=corr['pairs'],
                dbound=self.img_view_transformer.grid_config['depth'][:2])
            if rgb_s is not None and rgb_s.shape[2] > 1 and rgb_t.shape[2] > 0:
                losses['loss_rgb'] = self.get_rgb_loss(
                    rgb_s, rgb_t, weight=self.photo_loss_weight)
        losses.update(self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d,
                                             img_metas, gt_bboxes_ignore))
        return losses
