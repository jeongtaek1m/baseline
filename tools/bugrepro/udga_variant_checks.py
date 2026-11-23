"""CPU checks of the two extra UDGA variants (decision 2026-11-23, docs/decisions.md).

    cd bevdepth && CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/gen2e-bevdepth/bin/python -I \
        ../tools/bugrepro/udga_variant_checks.py [names]

ld-target   UDGA-Ld: LiDAR depth d -> virtual bin (get_depth_loss_virtual: v = d * k, vanilla
            labelling) -> the lift's metric depth of that bin ((d0 + b * step) / k). Must come
            back within one bin (step / k). The official target d / k is lifted to d / k^2.
lp-identity UDGA-Lp: warping a camera onto itself must give zero photometric error.
lp-depth    UDGA-Lp: on real R1 images, the photometric error over overlap pixels must be
            smallest at the true depth (LiDAR, nearest-neighbour filled) among scaled depths
            (x0.7 .. x1.3) and a constant 10 m. Run on a test-time sample and on an augmented
            training sample (resize, crop, flip, rotation, BDA) to cover post_rots/post_trans.
lp-grad     UDGA-Lp: photometric_loss_paper gives the depth logits a finite, non-zero gradient.
"""
import copy
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
BEVDEPTH = os.path.join(REPO, 'bevdepth')
CFG = 'configs/gen2e/udga-r50-pccr-Ld.py'


def sample(train, index=0, seed=0):
    """img_inputs (with bda) and metric LiDAR depth maps of one R1 sample on CPU."""
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    cfg = Config.fromfile(CFG)
    d = copy.deepcopy(cfg.data.train)
    d.ann_file = '../data/pccr/R1/R1_infos_mini.pkl'
    keep = ('PrepareImageInputsPCCR', 'LoadAnnotationsBEVDepthExtAug', 'LoadPointsFromFile',
            'PointToMultiViewDepthUpstream')
    pipe = [copy.deepcopy(p) for p in cfg.train_pipeline if p['type'] in keep]
    if not train:
        pipe[0]['is_train'] = False
        pipe[1] = dict(type='LoadAnnotationsBEVDepthUpstream', bda_aug_conf=cfg.bda_aug_conf,
                       classes=cfg.class_names, is_train=False)
    d.pipeline = pipe
    ds = build_dataset(d)
    np.random.seed(seed)
    info = ds.get_data_info(index)
    ds.pre_pipeline(info)
    r = ds.pipeline(info)
    return [t[None] for t in r['img_inputs']], r['gt_depth'][None], cfg


def model_parts(cfg):
    from mmdet3d.models import build_model
    from mmdet3d.models.detectors import bevdepth_udga as U
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    return model.img_view_transformer, U, model.compute_ssim_loss


def check_ld_target():
    (imgs, rots, trans, intrins, post_rots, post_trans, bda), gt, cfg = sample(train=True, seed=1)
    vt, _, _ = model_parts(cfg)
    k = vt.depth_scale(intrins, post_rots)  # (1, N)
    d0, _, step = vt.grid_config['depth']
    errs, offs = [], []
    for n in range(gt.shape[1]):
        d = gt[0, n][gt[0, n] > 0]
        v = d * k[0, n]
        b = torch.floor((v - (d0 - step)) / step) - 1  # vanilla labelling (get_downsampled_gt_depth)
        ok = (b >= 0) & (b < vt.D)
        lift = (d0 + b[ok] * step) / k[0, n]  # get_lidar_coor's metric depth of bin b
        errs.append((lift - d[ok]).abs().max().item() * k[0, n].item() / step)  # in bins
        off = d[ok] / k[0, n] ** 2  # official: target d / k, lifted as / k again
        offs.append((off - d[ok]).abs().max().item())
    ok = max(errs) < 1.0 + 1e-4
    print(f'ld-target   {"PASS" if ok else "FAIL"}  k per camera {k[0].numpy().round(3)}; '
          f'|lift(bin(d)) - d| <= {max(errs):.3f} bin (must be < 1); official d/k target would be '
          f'lifted up to {max(offs):.1f} m away')
    return ok


def check_lp_identity():
    (imgs, rots, trans, intrins, post_rots, post_trans, bda), gt, cfg = sample(train=True, seed=2)
    vt, U, ssim = model_parts(cfg)
    rgb = ((imgs * torch.tensor(U.IMG_STD).view(1, 1, 3, 1, 1) + torch.tensor(U.IMG_MEAN).view(1, 1, 3, 1, 1))
           / 255).clamp(0, 1)
    depth = torch.full(rgb.shape[:2] + rgb.shape[-2:], 15.0)
    maps = U.photometric_error_maps(depth, rgb, rots, trans, intrins, post_rots, post_trans,
                                    [[(n, n) for n in range(rgb.shape[1])]], ssim)
    worst = max(pe[v].abs().max().item() for pe, v in maps)
    frac = np.mean([v.float().mean().item() for _, v in maps])
    ok = worst < 1e-3
    print(f'lp-identity {"PASS" if ok else "FAIL"}  self-warp error max {worst:.2e} over valid pixels '
          f'(valid fraction {frac:.3f}, augmentation padding excluded)')
    return ok


def _dense(depth):
    """Nearest-neighbour fill of a sparse depth map; also returns the mask of real points."""
    from scipy import ndimage
    d = depth.numpy()
    m = d > 0
    idx = ndimage.distance_transform_edt(~m, return_distances=False, return_indices=True)
    return torch.from_numpy(d[tuple(idx)]), torch.from_numpy(m)


def check_lp_depth():
    ok_all = True
    for train in (False, True):
        (imgs, rots, trans, intrins, post_rots, post_trans, bda), gt, cfg = sample(train=train, seed=3)
        vt, U, ssim = model_parts(cfg)
        from mmdet3d.models.detectors.bevdepth_udga import frustum_correspondences
        corr = frustum_correspondences(vt, rots, trans, intrins, post_rots, post_trans)
        rgb = ((imgs * torch.tensor(U.IMG_STD).view(1, 1, 3, 1, 1)
                + torch.tensor(U.IMG_MEAN).view(1, 1, 3, 1, 1)) / 255).clamp(0, 1)
        dense, real = zip(*[_dense(gt[0, n]) for n in range(gt.shape[1])])
        dense = torch.stack(dense)[None]
        real = torch.stack(real)
        res = {}
        for name, dep in [('x0.70', dense * 0.7), ('x0.85', dense * 0.85), ('x1.00', dense),
                          ('x1.15', dense * 1.15), ('x1.30', dense * 1.3),
                          ('10 m', torch.full_like(dense, 10.0))]:
            maps = U.photometric_error_maps(dep, rgb, rots, trans, intrins, post_rots, post_trans,
                                            corr['pairs'], ssim)
            vals = []
            for (pe, v), (i, j) in zip(maps, corr['pairs'][0]):
                m = v & real[i]
                if m.any():
                    vals.append(pe[m])
            res[name] = torch.cat(vals).mean().item()
        best = min(res, key=res.get)
        ok = best == 'x1.00'
        ok_all &= ok
        tag = 'train sample (augmented)' if train else 'test-time sample'
        print(f'lp-depth    {"PASS" if ok else "FAIL"}  {tag}, {len(corr["pairs"][0])} pairs: mean pe at LiDAR '
              f'pixels ' + ', '.join(f'{k} {v:.4f}' for k, v in res.items()) + f' -> lowest {best}')
    return ok_all


def check_lp_grad():
    (imgs, rots, trans, intrins, post_rots, post_trans, bda), gt, cfg = sample(train=True, seed=4)
    vt, U, ssim = model_parts(cfg)
    from mmdet3d.models.detectors.bevdepth_udga import frustum_correspondences
    corr = frustum_correspondences(vt, rots, trans, intrins, post_rots, post_trans)
    N = imgs.shape[1]
    logits = torch.randn(N, vt.D, 24, 44, requires_grad=True)
    loss = U.photometric_loss_paper(logits, imgs, rots, trans, intrins, post_rots, post_trans,
                                    corr['pairs'], vt, ssim, weight=2.0)
    loss.backward()
    g = logits.grad
    nz = (g.abs().sum(1) > 0).float().mean().item()
    ok = bool(torch.isfinite(loss)) and bool(torch.isfinite(g).all()) and nz > 0
    print(f'lp-grad     {"PASS" if ok else "FAIL"}  loss {loss.item():.4f}, gradient finite, '
          f'{100 * nz:.1f}% of feature pixels get a gradient (overlap columns), |grad| max {g.abs().max():.2e}')
    return ok


CHECKS = {'ld-target': check_ld_target, 'lp-identity': check_lp_identity,
          'lp-depth': check_lp_depth, 'lp-grad': check_lp_grad}


def main(names):
    os.chdir(BEVDEPTH)
    sys.path.insert(0, BEVDEPTH)
    torch.manual_seed(0)
    fails = sum(not CHECKS[n]() for n in (names or CHECKS))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
