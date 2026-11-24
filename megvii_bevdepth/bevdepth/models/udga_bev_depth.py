# UDGA port (Gen2E): new file. BEVDepth (Megvii) + UDGA domain-generalisation losses.
# UDGA is Samsung-proprietary; this is a re-implementation and stays in the private repository. References are to
# third_party/UDGA (@9198218); the functions are ported from our old-base re-implementation
# (bevdepth/mmdet3d/models/detectors/bevdepth_udga.py, docs/bug_ledger.md UDGA-1 ... 9) with the Megvii matrices.
"""UDGA pre-training stage (paper sec. 3.3, 3.5; config bevssl3sup_lyf_to_nus_cls1_d60_pc400_L1_extaug.py) on the
Megvii BEVDepth model, decisions 2026-11-22 / 2026-11-23 / 2026-11-24 (docs/decisions.md):

    L = L_det + overlap_loss_weight * L_ov + L_p [+ L_d]

- L_ov: the depth logits of camera i moved to the corresponding frustum cells of every overlapping camera j
  (frustum_correspondences), averaged per target cell, SmoothL1 against camera j's logits there (weight 1);
- L_p, photo_loss_mode: 'public' = the released term (photo_reprojection_public: argmax depth, no gradient, UDGA-4
  kept), 'paper' = paper eq. 4 with bilinear sampling (photometric_loss_paper, differentiable); weight 2.0;
- L_d (the exp's lidar_depth flag): the Megvii base's own depth loss on virtual-depth targets d * k (UDGA-Ld);
- virtual depth bins: UDGALSSFPN (pixel_const 400); ext aug: PCCRUDGADataset.
Inference is the plain BEVDepth path. LEDA (adapters) belongs to the label-efficient adaptation stage, not to DG.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from bevdepth.layers.backbones.udga_lss_fpn import UDGALSSFPN
from bevdepth.layers.heads.bev_depth_head import BEVDepthHead

__all__ = ['UDGABEVDepth', 'frustum_correspondences', 'overlap_loss', 'bevdet_cams']

# the Megvii dataset's mmcv.imnormalize (img_conf): (x - mean) / std on 0..255 values (channel order: see to_rgb)
IMG_MEAN = (123.675, 116.28, 103.53)
IMG_STD = (58.395, 57.12, 57.375)


def bevdet_cams(mats_dict):
    """Key-frame cameras in BEVDet form: rots / trans (camera -> ego), intrins, post_rots / post_trans (ida)."""
    s2e = mats_dict['sensor2ego_mats'][:, 0]
    intrin = mats_dict['intrin_mats'][:, 0]
    ida = mats_dict['ida_mats'][:, 0]
    return s2e[..., :3, :3], s2e[..., :3, 3], intrin[..., :3, :3], ida[..., :3, :3], ida[..., :3, 3]


@torch.no_grad()
def frustum_correspondences(backbone, mats_dict):
    """Corresponding frustum cells between all ordered camera pairs of each sample.

    Re-implements the geometry of UDGA's overlap constraint (BEVSSL3.reprojection, bevssl.py:51-129) with the bugs
    fixed: the cell's 3D point is the lift's own frustum point, in the logits' memory order (UDGA-1, UDGA-3: metric
    depth = virtual depth / k); no hard-coded pairs, every ordered pair (i, j) with at least one point of camera i in
    camera j's frustum (UDGA-7); B and N from the tensors (UDGA-8). A point corresponds to camera j's nearest
    frustum pixel and to the virtual-depth bin [d0 + b * step, d0 + (b + 1) * step) holding its virtual depth in
    camera j (BEVDepth's label convention). No gradient flows through the geometry.

    Returns flat indices into logits.reshape(-1), logits (B * N, D, fH, fW):
      src (P,) source cells; dst (P,) the target cell of each point; key (P,) one id per (sample, pair, target cell),
      points sharing a key are averaged (UDGA-6); pairs: list over samples of the ordered pairs (i, j).
    """
    rots, trans, intrins, post_rots, post_trans = bevdet_cams(mats_dict)
    s2e, intrin, ida = (mats_dict[k][:, 0] for k in ('sensor2ego_mats', 'intrin_mats', 'ida_mats'))
    B, N = trans.shape[:2]
    D, fH, fW, _ = backbone.frustum.shape
    d0, _, step = backbone.d_bound
    pts = backbone.get_geometry(s2e, intrin, ida, None)  # (B, N, D, fH, fW, 3), ego frame without bda
    k = backbone.depth_scale(intrin, ida)  # (B, N)
    # image -> feature grid: the frustum samples pixel x = c * (W_in - 1) / (fW - 1)
    w_in1 = float(backbone.frustum[0, 0, -1, 0])
    h_in1 = float(backbone.frustum[0, -1, 0, 1])
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
            ok = ((z > 1e-3) & (col >= 0) & (col < fW) & (row >= 0) & (row < fH) & (dbin >= 0) & (dbin < D)
                  & (src_cam.reshape(-1) != j))
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
    """UDGA's multi-view overlap depth constraint L_ov (paper eq. 2-3), corresponding cells only: values landing in
    the same cell are averaged (UDGA-6), SmoothL1 on those cells only (UDGA-9); logits, beta 1 and no detach as the
    official config; an empty set gives 0 instead of NaN (UDGA-6)."""
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
        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


@torch.no_grad()
def photo_reprojection_public(depth_pred, img, rots, trans, intrins, post_rots, post_trans, pairs, dbound,
                              downsample=16):
    """UDGA BEVSSL3.photo_reprojection (bevssl.py:170-223), kept as the public code (decision 2026-11-23): no gradient
    (argmax depth, integer indexing; UDGA-4), the argmax bin index used as metric depth, the (w, h)-ordered pixel grid
    paired with the (h, w)-ordered depth (the UDGA-1 transposition). Only what PCCR needs changed: `pairs` replaces
    the hard-coded 6-camera table (UDGA-7); B, N come from the tensors (UDGA-8)."""
    device = depth_pred.device
    B, N, _, H, W = img.shape
    BN, D, height, width = depth_pred.shape
    depth = torch.argmax(depth_pred.sigmoid(), dim=1)

    coor_w = torch.linspace(0, width - 1, width)
    coor_h = torch.linspace(0, height - 1, height)
    coor_w, coor_h = torch.meshgrid(coor_w, coor_h, indexing='ij')
    wh = torch.stack([coor_w * downsample, coor_h * downsample], dim=-1).expand(B, N, width, height, 2).reshape(
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
            adj_depth = torch.cat([adj_depth[:, :2] / adj_depth[:, 2:3], adj_depth[:, 2:3]], 1)
            adj_depth = adj_depth.matmul(post_rots[b, pid].T) + post_trans[b, pid:pid + 1, :]
            adj_depth[:, :2] = torch.round(adj_depth[:, :2])
            kept1 = (adj_depth[:, 0] >= 0) & (adj_depth[:, 0] < W) & (adj_depth[:, 1] >= 0) \
                & (adj_depth[:, 1] < H) & (adj_depth[:, 2] < dbound[1]) & (adj_depth[:, 2] >= dbound[0])
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


def expected_metric_depth(depth_logits, k, d_bound, out_hw):
    """Expected metric depth of every image pixel, (B, N, H, W), differentiable: E[v] over the softmax of the virtual
    bins (bin b at d0 + b * step, as the lift places it), divided by k, bilinearly upsampled with align_corners=True
    (the frustum's pixel grid linspace(0, W_in - 1, fW))."""
    B, N = k.shape
    BN, D, fh, fw = depth_logits.shape
    d0, _, step = d_bound
    bins = d0 + step * torch.arange(D, device=depth_logits.device, dtype=depth_logits.dtype)
    ev = (depth_logits.softmax(1) * bins.view(1, D, 1, 1)).sum(1, keepdim=True)
    em = ev / k.reshape(BN, 1, 1, 1)
    return F.interpolate(em, size=out_hw, mode='bilinear', align_corners=True).view(B, N, *out_hw)


def photometric_error_maps(depth, rgb, rots, trans, intrins, post_rots, post_trans, pairs, ssim):
    """Per-pair photometric error of UDGA's L_p (paper eq. 4), differentiable in `depth` (old-base function).

    depth (B, N, H, W): metric depth of camera i's pixels; rgb (B, N, 3, H, W) in [0, 1]. Every pixel of camera i is
    back-projected with its depth, projected into camera j and I_j is bilinearly sampled there; the error is the
    public mix pe = 0.85 * (1 - SSIM) / 2 + 0.15 * |I_i - I_j<p*>| (bevssl.py:225-233). A pixel counts when its
    projection lies inside image j in front of the camera and neither image is augmentation padding (black) there.
    Returns a list over (b, i, j) of (pe (H, W), valid (H, W) bool)."""
    B, N, _, H, W = rgb.shape
    dev, dt = depth.device, depth.dtype
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=dt), torch.arange(W, device=dev, dtype=dt),
                            indexing='ij')
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
            warped = F.grid_sample(rgb[b, j:j + 1], grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            warped_content = F.grid_sample(content[b, j:j + 1], grid, mode='bilinear', padding_mode='zeros',
                                           align_corners=True)
            target = rgb[b, i:i + 1]
            l1 = (warped - target).abs().mean(1)[0]
            ssim_map = ssim(warped, target).mean(1)[0]
            pe = 0.85 * ssim_map + 0.15 * l1
            valid = ((z.view(H, W) > 1e-3) & (gx.view(H, W).abs() <= 1) & (gy.view(H, W).abs() <= 1)
                     & (warped_content[0, 0] > 0.999) & (content[b, i, 0] > 0.5)).detach()
            out.append((pe, valid))
    return out


def to_unit_rgb(imgs):
    """The dataset's normalised images back to [0, 1] (same channel order as the network input)."""
    mean = imgs.new_tensor(IMG_MEAN).view(1, 1, 3, 1, 1)
    std = imgs.new_tensor(IMG_STD).view(1, 1, 3, 1, 1)
    return ((imgs * std + mean) / 255.0).clamp(0, 1)


def photometric_loss_paper(depth_logits, imgs, cams, pairs, k, d_bound, ssim, weight=2.0):
    """UDGA L_p as the paper writes it (eq. 4), with a gradient (UDGA-4 fixed): expected metric depth, [0, 1] images,
    photometric_error_maps over the overlapping pairs, mean over the valid pixels of the batch, times `weight`."""
    H, W = imgs.shape[-2:]
    rgb = to_unit_rgb(imgs)
    depth = expected_metric_depth(depth_logits, k, d_bound, (H, W))
    maps = photometric_error_maps(depth, rgb, *cams, pairs, ssim)
    vals = [pe[v] for pe, v in maps if bool(v.any())]
    if not vals:
        return depth_logits.sum() * 0.0
    return weight * torch.cat(vals).mean()


class UDGABEVDepth(nn.Module):

    def __init__(self, backbone_conf, head_conf, pixel_const=400.0, overlap_loss_weight=1.0, overlap_loss_beta=1.0,
                 photo_loss_weight=2.0, photo_loss_mode='public'):
        super().__init__()
        assert photo_loss_mode in ('public', 'paper')
        self.backbone = UDGALSSFPN(pixel_const=pixel_const, **backbone_conf)
        self.head = BEVDepthHead(**head_conf)
        self.overlap_loss_weight = overlap_loss_weight
        self.overlap_loss_beta = overlap_loss_beta
        self.photo_loss_weight = photo_loss_weight
        self.photo_loss_mode = photo_loss_mode
        self.compute_ssim_loss = SSIM()

    def forward(self, x, mats_dict, timestamps=None):
        """Inference: the plain BEVDepth path."""
        return self.head(self.backbone(x, mats_dict, timestamps))

    def get_rgb_loss(self, rgb1, rgb2, weight=2.0):
        """UDGA BEVSSL3.get_rgb_loss (bevssl.py:225-233)."""
        diff_img = (rgb1 - rgb2).abs()
        diff_img = diff_img.clamp(0, 1)
        ssim_map = self.compute_ssim_loss(rgb1, rgb2)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)
        b, h, w, c = diff_img.shape
        return weight * diff_img.sum() / (b * h * w * c)

    def forward_train(self, x, mats_dict):
        """Head predictions, depth logits (B * N, D, fH, fW) and the UDGA losses (L_ov, L_p)."""
        bev, depth_logits = self.backbone(x, mats_dict, return_logits=True)
        preds = self.head(bev)
        cams = bevdet_cams(mats_dict)
        corr = frustum_correspondences(self.backbone, mats_dict)
        losses = dict(loss_overlap=self.overlap_loss_weight *
                      overlap_loss(depth_logits, corr, beta=self.overlap_loss_beta))
        imgs = x[:, 0]
        if self.photo_loss_weight > 0 and self.photo_loss_mode == 'paper':
            k = self.backbone.depth_scale(mats_dict['intrin_mats'][:, 0], mats_dict['ida_mats'][:, 0])
            losses['loss_rgb'] = photometric_loss_paper(depth_logits, imgs, cams, corr['pairs'], k,
                                                        self.backbone.d_bound, self.compute_ssim_loss,
                                                        weight=self.photo_loss_weight)
        elif self.photo_loss_weight > 0:
            rgb_s, rgb_t = photo_reprojection_public(depth_logits, imgs, *cams, pairs=corr['pairs'],
                                                     dbound=self.backbone.d_bound[:2])
            if rgb_s is not None and rgb_s.shape[2] > 1 and rgb_t.shape[2] > 0:
                losses['loss_rgb'] = self.get_rgb_loss(rgb_s, rgb_t, weight=self.photo_loss_weight)
        return preds, depth_logits, losses, corr

    def get_targets(self, gt_boxes, gt_labels):
        return self.head.get_targets(gt_boxes, gt_labels)

    def loss(self, targets, preds_dicts):
        return self.head.loss(targets, preds_dicts)

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)
