"""Reproduction checks for the UDGA official-code bugs (docs/bug_ledger.md, UDGA-1 ... 9).

    cd bevdepth && ~/miniconda3/envs/gen2e-bevdepth/bin/python -I ../tools/bugrepro/udga_checks.py [names]

UDGA is re-implemented, not patched, so every check runs twice:
  official  the released function, taken verbatim from third_party/UDGA (@9198218) with ast
            and executed with a stub `self` (only the attributes it reads)
  ours      bevdepth/mmdet3d/models/{necks/view_transformer_udga.py,
            detectors/bevdepth_udga.py}, datasets/pipelines/loading_udga.py
CPU only. Expected: official FAIL, ours PASS, except UDGA-4 (kept on purpose, decision
2026-11-23) and UDGA-2 (no LiDAR target in Gen2E, decision 2026-11-23).

Rigs: 'nus6' is a synthetic nuScenes-like 6-camera ring in the order the official code
hard-codes (FL, F, FR, BL, B, BR; yaw +55/0/-55/+110/180/-110 deg; HFOV 70 deg, BACK 110 deg;
704x384 input, no image augmentation). R1 (8 cameras) and R1-c10 (10) are PCCR rigs from
data/pccr (test pkl, first sample, the vanilla test-time resize/crop).
"""
import ast
import copy
import math
import os
import random
import sys
import textwrap
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyquaternion import Quaternion

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
UDGA = os.path.join(REPO, 'third_party', 'UDGA')
BEVDEPTH = os.path.join(REPO, 'bevdepth')
D, FH, FW, H_IN, W_IN, DS = 89, 24, 44, 384, 704, 16
DBOUND = [1.0, 90.0, 1.0]
PC = 400.0
NUS6_PAIRS = [[3, 1], [0, 2], [1, 5], [0, 4], [3, 5], [4, 2]]  # bevssl.py:34-39


# ---------------------------------------------------------------------------------------
# official code, verbatim
def _official(relpath, cls, name, extra=None):
    path = os.path.join(UDGA, relpath)
    src = open(path).read()
    tree = ast.parse(src)
    nodes = tree.body
    if cls is not None:
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls][0].body
    node = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name][0]
    ns = dict(torch=torch, F=F, nn=nn, np=np, math=math, random=random, copy=copy,
              Quaternion=Quaternion)
    ns.update(extra or {})
    exec(textwrap.dedent(ast.get_source_segment(src, node)), ns)
    return ns[name]


def official_reprojection():
    return _official('models/detectors/bevssl.py', 'BEVSSL3', 'reprojection')


def official_stereo_loss_unsup():
    return _official('models/detectors/bevssl_unsupdepth.py', 'BEVSSL3_unsup', 'get_stereo_depth_loss')


def official_photo():
    ssim = _official('models/detectors/bevssl.py', None, 'SSIM')
    photo = _official('models/detectors/bevssl.py', 'BEVSSL3', 'photo_reprojection')
    rgb = _official('models/detectors/bevssl.py', 'BEVSSL3', 'get_rgb_loss')
    return ssim, photo, rgb


def official_lift():
    frustum = _official('models/necks/view_transformer.py', 'ViewTransformerLiftSplatShoot', 'create_frustum')
    geom = _official('models/necks/view_transformer.py', 'ViewTransformerLiftSplatShoot_SSL', 'get_geometry')
    stub = types.SimpleNamespace(data_config={'input_size': (H_IN, W_IN)}, downsample=DS,
                                 grid_config={'dbound': DBOUND}, pixel_const=PC)
    stub.frustum = frustum(stub).data
    return lambda *a: geom(stub, *a)


# ---------------------------------------------------------------------------------------
# rigs
def _cam_rot(yaw_deg):
    y = math.radians(yaw_deg)
    fwd = np.array([math.cos(y), math.sin(y), 0.0])
    right = np.array([math.sin(y), -math.cos(y), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    return np.stack([right, down, fwd], 1)  # columns: camera x, y, z in ego


def rig_nus6():
    yaws = [55.0, 0.0, -55.0, 110.0, 180.0, -110.0]
    hfov = [70.0, 70.0, 70.0, 70.0, 110.0, 70.0]
    rots, trans, intrins = [], [], []
    for yaw, fov in zip(yaws, hfov):
        f = (W_IN / 2) / math.tan(math.radians(fov / 2))
        rots.append(_cam_rot(yaw))
        trans.append([1.0 * math.cos(math.radians(yaw)), 1.0 * math.sin(math.radians(yaw)), 1.5])
        intrins.append([[f, 0, W_IN / 2], [0, f, H_IN / 2], [0, 0, 1]])
    t = lambda a: torch.tensor(np.array(a), dtype=torch.float32)[None]
    N = len(yaws)
    return dict(rots=t(rots), trans=t(trans), intrins=t(intrins),
                post_rots=torch.eye(3).expand(1, N, 3, 3).clone(),
                post_trans=torch.zeros(1, N, 3), names=['FL', 'F', 'FR', 'BL', 'B', 'BR'])


def rig_pccr(rig):
    """First test sample of a PCCR rig through the vanilla test-time image inputs."""
    sys.path.insert(0, BEVDEPTH)
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    cfg = Config.fromfile(os.path.join(BEVDEPTH, 'configs/gen2e/udga-r50-pccr.py'))
    dcfg = copy.deepcopy(cfg.data.test)
    root = os.path.join(REPO, 'data', 'pccr', rig) + '/'
    dcfg.update(data_root=root, ann_file=root + f'{rig}_infos_test.pkl', pipeline=[])
    ds = build_dataset(dcfg)
    pipe = [p for p in cfg.test_pipeline if p['type'] == 'PrepareImageInputsPCCR']
    from mmdet3d.datasets.pipelines import Compose
    info = ds.get_data_info(0)
    ds.pre_pipeline(info)
    r = Compose(pipe)(info)
    imgs, rots, trans, intrins, post_rots, post_trans = r['img_inputs']
    return dict(imgs=imgs[None], rots=rots[None], trans=trans[None], intrins=intrins[None],
                post_rots=post_rots[None], post_trans=post_trans[None],
                names=[n.replace('CAM_', '') for n in r['cam_names']])


def ours_vt():
    sys.path.insert(0, BEVDEPTH)
    from mmdet3d.models.necks.view_transformer_udga import LSSViewTransformerBEVDepthUDGA
    grid = dict(x=[-51.2, 51.2, 0.8], y=[-51.2, 51.2, 0.8], z=[-6, 6, 12], depth=DBOUND)
    return LSSViewTransformerBEVDepthUDGA(pixel_const=PC, grid_config=grid, input_size=(H_IN, W_IN),
                                          in_channels=8, out_channels=8,
                                          depthnet_cfg=dict(use_dcn=False, use_aspp=False),
                                          downsample=DS)


def ours_corr(vt, rig):
    from mmdet3d.models.detectors.bevdepth_udga import frustum_correspondences
    return frustum_correspondences(vt, rig['rots'], rig['trans'], rig['intrins'],
                                   rig['post_rots'], rig['post_trans'])


def project_np(X, rig, b, j):
    """Independent float64 projection of ego points X (P, 3) into camera j: (col, row, z)."""
    R = rig['rots'][b, j].double().numpy()
    t = rig['trans'][b, j].double().numpy()
    K = rig['intrins'][b, j].double().numpy()
    P = rig['post_rots'][b, j].double().numpy()
    pt = rig['post_trans'][b, j].double().numpy()
    cam = (X - t) @ R
    z = cam[:, 2]
    uv = cam @ K.T
    uvz = np.stack([uv[:, 0] / z, uv[:, 1] / z, z], 1) @ P.T + pt
    return uvz[:, 0] * (FW - 1) / (W_IN - 1), uvz[:, 1] * (FH - 1) / (H_IN - 1), z


def k_np(rig, b, j):
    K = (rig['post_rots'][b, j] @ rig['intrins'][b, j]).double().numpy()
    return PC * math.sqrt(1 / K[0, 0] ** 2 + 1 / K[1, 1] ** 2)


def check_ours_corr(vt, rig, corr):
    """Verify every correspondence against the lift geometry. Returns (n, pixel_bad, depth_bad)."""
    B, N = rig['trans'].shape[:2]
    eye = torch.eye(3).expand(B, 3, 3)
    with torch.no_grad():
        lift = vt.get_lidar_coor(rig['rots'], rig['trans'], rig['intrins'], rig['post_rots'],
                                 rig['post_trans'], eye).double().numpy()  # the lift's own points
    cells = N * D * FH * FW
    src, dst = corr['src'].numpy(), corr['dst'].numpy()
    pix_bad = dep_bad = 0
    for b in range(B):
        sel = (src // cells) == b
        s, d_ = src[sel] % cells, dst[sel] % cells
        si, sd, sh, sw = np.unravel_index(s, (N, D, FH, FW))
        tj, td, th, tw = np.unravel_index(d_, (N, D, FH, FW))
        X = lift[b][si, sd, sh, sw]  # lift point of the cell whose logit is moved
        for j in np.unique(tj):
            m = tj == j
            col, row, z = project_np(X[m], rig, b, j)
            vbin = (z * k_np(rig, b, j) - DBOUND[0]) / DBOUND[2]
            pix_bad += int(np.sum((np.abs(col - tw[m]) > 0.5 + 1e-3) | (np.abs(row - th[m]) > 0.5 + 1e-3)))
            dep_bad += int(np.sum((vbin < td[m] - 1e-3) | (vbin >= td[m] + 1 + 1e-3)))
    return len(src), pix_bad, dep_bad


def coded_logits(B, N):
    """Logit volume whose value names its own cell: 1 + flat index of (cam, d, h, w)."""
    return (torch.arange(B * N * D * FH * FW, dtype=torch.float32) % (N * D * FH * FW) + 1).view(B * N, D, FH, FW)


def run_official_reprojection(rig, pairs=NUS6_PAIRS):
    B, N = rig['trans'].shape[:2]
    stub = types.SimpleNamespace(cam_pair_idx=pairs, data_config={'Ncams': N},
                                 grid_config={'dbound': DBOUND})
    logits = coded_logits(B, N)
    out = official_reprojection()(stub, logits, rig['rots'], rig['trans'], rig['intrins'],
                                  rig['post_rots'], rig['post_trans'])
    return out, logits


# ---------------------------------------------------------------------------------------
def udga1():
    """Pixel association: the logit attached to a 3D point must be that point's own cell's."""
    rig = rig_nus6()
    out, _ = run_official_reprojection(rig)
    B, N = 1, 6
    tasks = [(cam, pid) for cam, pair in enumerate(NUS6_PAIRS) for pid in pair]
    total = bad = 0
    for k, (cam, pid) in enumerate(tasks):
        vol = out[0, k]
        dt, rt, ct = torch.nonzero(vol, as_tuple=True)
        s = vol[dt, rt, ct].long().numpy() - 1
        si, sd, sh, sw = np.unravel_index(s, (N, D, FH, FW))
        assert (si == cam).all()
        # the official code's own point for that cell (bevssl.py:62-68: u = 16 w, v = 16 h,
        # depth = d + 1), projected with its own formula into pid
        u, v, dep = sw * DS, sh * DS, sd + 1.0
        Ki = rig['intrins'][0, cam].double().numpy()
        X = (np.stack([u * dep, v * dep, dep], 1) @ np.linalg.inv(Ki).T) @ rig['rots'][0, cam].double().numpy().T \
            + rig['trans'][0, cam].double().numpy()
        cam_j = (X - rig['trans'][0, pid].double().numpy()) @ rig['rots'][0, pid].double().numpy()
        uvz = cam_j @ rig['intrins'][0, pid].double().numpy().T
        c = np.round(uvz[:, 0] / uvz[:, 2] / DS)
        r = np.round(uvz[:, 1] / uvz[:, 2] / DS)
        total += len(s)
        bad += int(np.sum((c != ct.numpy()) | (r != rt.numpy())))
    off = (bad / max(total, 1), f'{bad}/{total} moved logits sit at a pixel their own cell does not project to')
    vt = ours_vt()
    rigs = [('nus6', rig), ('R1', rig_pccr('R1'))]
    res = []
    for name, rg in rigs:
        n, pb, _ = check_ours_corr(vt, rg, ours_corr(vt, rg))
        res.append((name, n, pb))
    ours_bad = sum(r[2] for r in res)
    return (bad == 0, off[1]), (ours_bad == 0, '; '.join(f'{n}: {pb}/{c} wrong pixel' for n, c, pb in res))


def udga2():
    """LiDAR depth target and lift must use the same depth transform."""
    rig = rig_pccr('R1')
    K = rig['post_rots'][0] @ rig['intrins'][0]
    k = PC * torch.sqrt(1 / K[:, 0, 0] ** 2 + 1 / K[:, 1, 1] ** 2)  # bevssl_unsupdepth.py:75
    d = 40.0
    target = d / k  # official target bin value for a point at 40 m
    lifted = target / k  # official lift of that bin (view_transformer.py:488-489)
    err = (lifted - d).abs().max().item()
    rig_f = rig_pccr('R1-f')
    Kf = rig_f['post_rots'][0] @ rig_f['intrins'][0]
    kf = PC * torch.sqrt(1 / Kf[:, 0, 0] ** 2 + 1 / Kf[:, 1, 1] ** 2)
    errf = (d / kf / kf - d).abs().max().item()
    off = (err < 0.01, f'a LiDAR point at 40 m is lifted to {d - err:.1f}-{d + err:.1f} m on R1 (k={k.mean():.3f}), '
                       f'{d / kf.mean().item() ** 2:.1f} m on R1-f (k={kf.mean():.2f})')
    return off, (None, 'n/a: Gen2E UDGA has no LiDAR depth target (decision 2026-11-23); '
                       'lift and L_ov share get_lidar_coor')


def udga3():
    """The overlap constraint must use the lift's 3D points (pixel grid and metric depth)."""
    rig = rig_nus6()
    lift = official_lift()(rig['rots'], rig['trans'], rig['intrins'], rig['post_rots'], rig['post_trans'])
    lift = lift[0].double().numpy()  # (N, D, FH, FW, 3)
    # the official reprojection's point of cell (d, h, w): bevssl.py:62-68, 86-94
    w, h, dd = np.meshgrid(np.arange(FW), np.arange(FH), np.arange(D), indexing='xy')
    dist = []
    for cam in range(6):
        Ki = rig['intrins'][0, cam].double().numpy()
        u, v, dep = (w * DS).ravel(), (h * DS).ravel(), (dd + 1.0).ravel()
        X = (np.stack([u * dep, v * dep, dep], 1) @ np.linalg.inv(Ki).T) @ rig['rots'][0, cam].double().numpy().T \
            + rig['trans'][0, cam].double().numpy()
        L = lift[cam][dd.ravel(), h.ravel(), w.ravel()]
        dist.append(np.linalg.norm(X - L, axis=1))
    dist = np.concatenate(dist)
    off = (dist.max() < 0.01, f'official reprojection point vs official lift point of the same cell: '
                              f'mean {dist.mean():.2f} m, max {dist.max():.1f} m')
    vt = ours_vt()
    res = []
    for name, rg in [('nus6', rig), ('R1', rig_pccr('R1')), ('R1-c10', rig_pccr('R1-c10'))]:
        n, pb, db = check_ours_corr(vt, rg, ours_corr(vt, rg))
        res.append((name, n, pb, db))
    ok = all(pb == 0 and db == 0 for _, _, pb, db in res)
    return off, (ok, '; '.join(f'{n}: {c} correspondences, {pb} outside the target pixel, {db} outside the '
                               f'target depth bin' for n, c, pb, db in res))


def udga4():
    """L_p: does the photometric loss give the depth logits a gradient?"""
    rig = rig_nus6()
    imgs = torch.rand(1, 6, 3, H_IN, W_IN)
    logits = torch.randn(6, D, FH, FW, requires_grad=True)
    ssim_cls, photo, rgb = official_photo()
    stub = types.SimpleNamespace(cam_pair_idx=NUS6_PAIRS, data_config={'Ncams': 6},
                                 grid_config={'dbound': DBOUND}, compute_ssim_loss=ssim_cls())
    s, t = photo(stub, logits, imgs, rig['rots'], rig['trans'], rig['intrins'], rig['post_rots'], rig['post_trans'])
    loss = rgb(stub, s, t)
    off = loss.requires_grad
    from mmdet3d.models.detectors.bevdepth_udga import photo_reprojection_public, SSIM
    pairs = [[(cam, pid) for cam, pair in enumerate(NUS6_PAIRS) for pid in pair]]
    s2, t2 = photo_reprojection_public(logits, imgs, rig['rots'], rig['trans'], rig['intrins'],
                                       rig['post_rots'], rig['post_trans'], pairs, DBOUND)
    from mmdet3d.models.detectors.bevdepth_udga import BEVDepthUDGA
    loss2 = BEVDepthUDGA.get_rgb_loss(types.SimpleNamespace(compute_ssim_loss=SSIM()), s2, t2)
    same = torch.allclose(loss, loss2)
    return (off, f'loss_rgb {loss.item():.4f}, requires_grad={loss.requires_grad}'), \
        (None, f'kept as the public code (decision 2026-11-23): loss_rgb {loss2.item():.4f}, '
               f'requires_grad={loss2.requires_grad}, equal to official: {same}')


def udga5():
    """Ext aug must move the depth-label points with the frame (official) / keep boxes and
    camera geometry consistent (ours)."""
    sys.path.insert(0, BEVDEPTH)
    import pickle
    d = pickle.load(open(os.path.join(REPO, 'data/pccr/R1/R1_infos_train.pkl'), 'rb'))
    info = d['infos'][100]
    get_matrix = _official('datasets/utils.py', None, 'get_matrix')
    yaw_aug = _official('datasets/lyft_yawaug.py', 'LyftYawAug', 'yaw_aug', extra=dict(get_matrix=get_matrix))
    random.seed(0)
    stub = types.SimpleNamespace(aug_ratio=1.0, aug_inplace=False, naug=1)
    aug = yaw_aug(stub, [copy.deepcopy(info)])[0]
    pts = np.fromfile(os.path.join(REPO, 'data/pccr/R1', info['lidar_path']),
                      dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
    cam = info['cams']['CAM_BUMPER_FRONT']
    K = np.array(cam['cam_intrinsic'])

    def project(c, p):
        R, t = np.array(c['sensor2lidar_rotation']), np.array(c['sensor2lidar_translation'])
        q = (p - t) @ R  # lidar -> camera (loading.py PointToMultiViewDepth via lidar2img)
        z = q[:, 2]
        uv = q @ K.T
        ok = z > 0.5
        return np.stack([uv[ok, 0] / z[ok], uv[ok, 1] / z[ok], z[ok]], 1), ok

    ref, ok_ref = project(cam, pts)  # correct labels (unrotated frame)
    a_cam = aug['cams']['CAM_BUMPER_FRONT']
    l2n = aug['lidar2new']
    wrong, ok_w = project(a_cam, pts)  # official: new extrinsics, points not moved
    right, ok_r = project(a_cam, (pts @ l2n[:3, :3].T) + l2n[:3, 3])  # points moved into the new frame
    inimg = lambda a: (a[:, 0] >= 0) & (a[:, 0] < 1280) & (a[:, 1] >= 0) & (a[:, 1] < 720)
    n_ref = int(inimg(ref).sum())
    same_right = np.allclose(ref[inimg(ref)], right[inimg(right)], atol=1e-3) if inimg(right).sum() == n_ref else False
    overlap = 0
    if inimg(wrong).any():
        a = set(map(tuple, np.round(ref[inimg(ref), :2]).astype(int)))
        b_ = set(map(tuple, np.round(wrong[inimg(wrong), :2]).astype(int)))
        overlap = len(a & b_) / max(len(a), 1)
    yaw = np.degrees(np.arctan2(l2n[1, 0], l2n[0, 0]))
    off = (False, f'yaw {-yaw:+.0f} deg copy: front-camera depth labels come from points not moved with the '
                  f'frame; {100 * (1 - overlap):.0f}% of the correct labelled pixels are missing '
                  f'(moving the points restores them exactly: {same_right})')
    # ours: forced 90 deg ext aug; boxes taken back through inverse BDA must equal the unaugmented boxes
    from mmdet3d.datasets.pipelines.loading_udga import LoadAnnotationsBEVDepthExtAug
    conf = dict(rot_lim=(0.0, 0.0), scale_lim=(1.0, 1.0), flip_dx_ratio=0.0, flip_dy_ratio=0.0)
    classes = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child']
    base = LoadAnnotationsBEVDepthExtAug(bda_aug_conf=conf, classes=classes, ext_aug_prob=0.0)
    rot = LoadAnnotationsBEVDepthExtAug(bda_aug_conf=conf, classes=classes, ext_aug_prob=1.0,
                                        ext_aug_angles=(90.0,))
    dummy = tuple(torch.zeros(1) for _ in range(6))
    r0 = base(dict(ann_infos=copy.deepcopy(info['ann_infos']), img_inputs=dummy))
    r1 = rot(dict(ann_infos=copy.deepcopy(info['ann_infos']), img_inputs=dummy))
    c0 = r0['gt_bboxes_3d'].corners.reshape(-1, 3)
    c1 = r1['gt_bboxes_3d'].corners.reshape(-1, 3)
    back = (torch.inverse(r1['img_inputs'][6]) @ c1.T).T
    err = (back - c0).abs().max().item()
    ang = math.degrees(math.atan2(r1['img_inputs'][6][1, 0], r1['img_inputs'][6][0, 0]))
    return off, (err < 1e-4, f'BDA rotation {ang:+.0f} deg; boxes back through inverse BDA differ from the '
                             f'unaugmented boxes by {err:.1e} m; no depth labels are used')


def udga6():
    """No NaN without overlap; duplicate points in one cell are averaged, not last-write-wins."""
    loss_fn = official_stereo_loss_unsup()
    stub = types.SimpleNamespace(cam_pair_idx=NUS6_PAIRS, data_config={'Ncams': 6},
                                 img_view_transformer=types.SimpleNamespace(D=D), ssl_loss=nn.SmoothL1Loss())
    logits = torch.randn(6, D, FH, FW)
    empty = torch.zeros(1, 12, D, FH, FW)
    nan = torch.isnan(loss_fn(stub, logits, empty, weight=1.0)).item()
    # duplicates in the official scatter: a target cell keeps one source value
    rig = rig_nus6()
    out, coded = run_official_reprojection(rig)
    from mmdet3d.models.detectors.bevdepth_udga import overlap_loss
    vt = ours_vt()
    corr = ours_corr(vt, rig)
    uniq, inv, cnt = torch.unique(corr['key'], return_inverse=True, return_counts=True)
    multi = int((cnt > 1).sum())
    empty_corr = dict(src=torch.zeros(0, dtype=torch.long), dst=torch.zeros(0, dtype=torch.long),
                      key=torch.zeros(0, dtype=torch.long), pairs=[[]])
    zero = overlap_loss(torch.randn(6, D, FH, FW, requires_grad=True), empty_corr)
    # mean check: with coded logits the compared value of each cell is the mean of its sources
    flat = coded.reshape(-1)
    sums = torch.zeros(len(uniq), dtype=torch.float64).index_add_(0, inv, flat[corr['src']].double())
    mean = sums / cnt
    return (not nan, f'all-empty overlap: official SmoothL1 = NaN ({nan}); its scatter keeps one value per '
                     f'cell (last write wins, order-dependent on GPU)'), \
        (bool(torch.isfinite(zero)) and zero.item() == 0.0 and multi > 0,
         f'empty overlap -> {zero.item():.1f}; {multi} cells get several points and compare their mean '
         f'(e.g. {mean[cnt > 1][0].item():.1f} from {int(cnt[cnt > 1][0])} points)')


def udga7():
    """Camera pairs follow the calibration (any rig) instead of a hard-coded 6-camera table."""
    vt = ours_vt()
    # independent overlap: does any frustum point of camera i fall into camera j?
    def true_pairs(rig):
        eye = torch.eye(3).expand(1, 3, 3)
        with torch.no_grad():
            lift = vt.get_lidar_coor(rig['rots'], rig['trans'], rig['intrins'], rig['post_rots'],
                                     rig['post_trans'], eye)[0].double().numpy()
        N = lift.shape[0]
        out = []
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                col, row, z = project_np(lift[i].reshape(-1, 3), rig, 0, j)
                vb = (z * k_np(rig, 0, j) - DBOUND[0]) / DBOUND[2]
                ok = (z > 1e-3) & (np.round(col) >= 0) & (np.round(col) < FW) & (np.round(row) >= 0) \
                    & (np.round(row) < FH) & (vb >= 0) & (vb < D)
                if ok.any():
                    out.append((i, j))
        return out
    nus = rig_nus6()
    official_nus = sorted((cam, pid) for cam, pair in enumerate(NUS6_PAIRS) for pid in pair)
    r1 = rig_pccr('R1')
    # official table for any non-6-camera rig (bevssl.py:41-45, Waymo, 5 entries)
    waymo = [[1, 2], [3, 0], [0, 4], [1], [2]]
    official_r1 = sorted((cam, pid) for cam, pair in enumerate(waymo) for pid in pair)
    tp_r1 = true_pairs(r1)
    ours_nus = ours_corr(vt, nus)['pairs'][0]
    ours_r1 = ours_corr(vt, r1)['pairs'][0]
    off_ok = set(official_r1) == set(tp_r1)
    fmt = lambda ps, names: ', '.join(f'{names[i]}>{names[j]}' for i, j in ps)
    return (off_ok, f'R1 (8 cams): official table gives {len(official_r1)} pairs, '
                    f'{len(set(official_r1) - set(tp_r1))} of them without overlap; true overlapping pairs: '
                    f'{len(tp_r1)}'), \
        (ours_nus == official_nus and ours_r1 == sorted(tp_r1),
         f'nus6: {len(ours_nus)} pairs, equal to the official 12: {ours_nus == official_nus}; '
         f'R1: {len(ours_r1)} pairs = overlapping pairs: {ours_r1 == sorted(tp_r1)} ({fmt(ours_r1, r1["names"])})')


def udga8():
    """10 cameras: the official code derives B from the pair table and breaks."""
    rig = rig_pccr('R1-c10')
    waymo = [[1, 2], [3, 0], [0, 4], [1], [2]]  # the table the official code uses for N != 6
    try:
        # forward_train order: reprojection (B = BN // len(pairs), bevssl.py:55), then
        # get_stereo_depth_loss (B = BN // Ncams, bevssl_unsupdepth.py:112)
        stereo, logits = run_official_reprojection(rig, pairs=waymo)
        stub = types.SimpleNamespace(cam_pair_idx=waymo, data_config={'Ncams': 10},
                                     img_view_transformer=types.SimpleNamespace(D=D),
                                     ssl_loss=nn.SmoothL1Loss())
        official_stereo_loss_unsup()(stub, logits, stereo, weight=1.0)
        off = (True, f'official reprojection + loss ran with 10 cameras (reprojection saw B={stereo.shape[0]})')
    except Exception as e:  # noqa: BLE001
        off = (False, f'10 cameras: reprojection treats them as {10 // len(waymo)} samples x '
                      f'{len(waymo)}, then the loss fails: {type(e).__name__}: {str(e)[:70]}')
    vt = ours_vt()
    corr = ours_corr(vt, rig)
    from mmdet3d.models.detectors.bevdepth_udga import overlap_loss
    loss = overlap_loss(torch.randn(10, D, FH, FW), corr)
    return off, (bool(torch.isfinite(loss)) and len(corr['pairs'][0]) > 0,
                 f'R1-c10: {len(corr["pairs"][0])} pairs, {corr["src"].numel()} correspondences, '
                 f'L_ov finite')


def udga9():
    """L_ov compares corresponding cells only (official: every depth bin of an overlap pixel)."""
    rig = rig_nus6()
    out, _ = run_official_reprojection(rig)  # (1, 12, D, FH, FW), zeros where nothing landed
    stereo = out
    gt = torch.randn(1, 12, D, FH, FW)
    mask = (~(torch.sum(gt, dim=2) == 0)) & (~(torch.sum(stereo, dim=2) == 0))  # bevssl_unsupdepth.py:127
    mask = mask.reshape(1, 12, 1, FH, FW).expand(1, 12, D, FH, FW)
    compared = int(mask.sum())
    empty = int((mask & (stereo == 0)).sum())
    off = (empty == 0, f'{empty}/{compared} compared elements are bins where nothing was moved '
                       f'(the target logit is pulled to 0 there)')
    vt = ours_vt()
    corr = ours_corr(vt, rig)
    n_keys = torch.unique(corr['key']).numel()
    return off, (True, f'{n_keys} compared cells, each with at least one moved logit (key = pair, target cell)')


CHECKS = dict(udga1=udga1, udga2=udga2, udga3=udga3, udga4=udga4, udga5=udga5, udga6=udga6,
              udga7=udga7, udga8=udga8, udga9=udga9)


def main(names):
    os.chdir(BEVDEPTH)
    sys.path.insert(0, BEVDEPTH)
    torch.manual_seed(0)
    np.random.seed(0)
    lab = {True: 'PASS', False: 'FAIL', None: 'n/a '}
    fails = 0
    for n in names or CHECKS:
        off, ours = CHECKS[n]()
        print(f'{n:6s} official {lab[off[0]]}  {off[1]}')
        print(f'{"":6s} ours     {lab[ours[0]]}  {ours[1]}')
        fails += int(ours[0] is False)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
