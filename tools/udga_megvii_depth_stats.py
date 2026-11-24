#!/usr/bin/env python3
"""Depth-distribution stats of a UDGA checkpoint (Megvii base) on a few test samples of one rig (GPU).

- max_prob: mean over all depth cells of the softmax maximum (a uniform distribution over the 89 bins gives 0.0112).
- spearman: rank correlation between the expected metric depth, sum_i p_i v_i / k, and the min-pooled LiDAR depth,
  over the cells that have LiDAR points.

Usage: python tools/udga_megvii_depth_stats.py CKPT [--rig R1] [--n 8]
Run with the gen2e-megvii python. Results for UDGA-Lp / -Ld are in docs/results.md (UDGA Megvii base section).
"""
import argparse
import os
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ckpt')
    ap.add_argument('--rig', default='R1')
    ap.add_argument('--n', type=int, default=8, help='test samples, evenly spaced over the split')
    args = ap.parse_args()
    ckpt = os.path.abspath(args.ckpt)  # check_udga_megvii changes the working directory on import
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import check_udga_megvii as C

    bb = C.backbone()
    sd = torch.load(ckpt, map_location='cpu')['state_dict']
    sd = {k[len('model.backbone.'):]: v for k, v in sd.items() if k.startswith('model.backbone.')}
    missing, unexpected = bb.load_state_dict(sd, strict=False)
    assert not unexpected and not [m for m in missing if 'num_batches' not in m], (missing, unexpected)
    bb = bb.cuda().eval()
    ds = C.make_ds(args.rig, 'test', False, return_depth=True, ext_aug_prob=0.0)
    v = torch.arange(*C.D_BOUND).cuda() + 0.0  # virtual depth of each bin
    max_prob, rho = [], []
    with torch.no_grad():
        for i in np.linspace(0, len(ds) - 1, args.n).astype(int):
            imgs, mats, _, _, _, _, depth = C.batch_of(ds, int(i), True)
            mats = {k: t.cuda() for k, t in mats.items()}
            imgs = imgs.cuda()
            B, _, N, _, H, _ = imgs.shape
            feats = bb.get_cam_feats(imgs)[:, 0]
            logits = bb._forward_depth_net(feats.reshape(B * N, *feats.shape[2:]), mats)[:, :bb.depth_channels]
            p = logits.softmax(1)  # (N, D, h, w)
            max_prob.append(float(p.max(1).values.mean()))
            k = bb.depth_scale(mats['intrin_mats'][:, 0], mats['ida_mats'][:, 0])[0]  # (N,)
            exp_d = (p * v.view(1, -1, 1, 1)).sum(1) / k.view(-1, 1, 1)  # expected metric depth, (N, h, w)
            h, w = exp_d.shape[1:]
            f = H // h
            gt = depth[0, 0].cuda().view(N, h, f, w, f).permute(0, 1, 3, 2, 4).reshape(N, h, w, f * f)
            gt = torch.where(gt > 0, gt, torch.full_like(gt, 1e5)).min(-1).values
            m = gt < 1e5
            a, b = exp_d[m].cpu().numpy(), gt[m].cpu().numpy()
            rho.append(float(np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1]))
    print(f'{ckpt}\n{args.rig}: {args.n} samples, max_prob {np.mean(max_prob):.4f} '
          f'(uniform {1 / len(v):.4f}), spearman {np.mean(rho):+.3f}')


if __name__ == '__main__':
    main()
