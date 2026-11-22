"""Reproduction checks for the PD-BEV bugs in docs/bug_ledger.md (CPU only).

    cd bevdepth && ~/miniconda3/envs/gen2e-bevdepth/bin/python ../tools/bugrepro/pdbev_checks.py [ID ...]

Each check prints PASS or FAIL with the measured numbers. They are written against the
behaviour each bug breaks, so they FAIL on the official code and PASS once the bug is fixed.
The ledger records the output before and after every fix. Checks on real data use the first
R1 mini samples (data/pccr/R1/R1_infos_mini.pkl) through the vanilla PCCR dataset adapter.

IDs: pdbev1 pdbev3 pdbev4 pdbev5 pdbev6 pdbev8 pdbev9 pdbev10 pdbev12 pdbev16 pdbev17
(pdbev6 also covers pdbev7; pdbev11 and pdbev15 are checked in the GPU smoke test.)
"""
import copy
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CLASSES = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child']
PC_RANGE = [-51.2, -51.2, -6.0, 51.2, 51.2, 6.0]


def report(name, ok, msg):
    print(f'{name:8s} {"PASS" if ok else "FAIL"}  {msg}')
    return ok


# ---------------------------------------------------------------- helpers on real data
_DATASET = None


def pccr_dataset():
    """Vanilla PCCR adapter on R1 mini (normalised paths, 7-class ann_infos labels)."""
    global _DATASET
    if _DATASET is None:
        from mmcv import Config
        from mmdet3d.datasets import build_dataset
        cfg = Config.fromfile(os.path.join(REPO, 'bevdepth/configs/gen2e/bevdepth-r50-pccr.py'))
        d = copy.deepcopy(cfg.data.train)
        d.update(pipeline=[], ann_file='../data/pccr/R1/R1_infos_mini.pkl', test_mode=False)
        _DATASET = build_dataset(d)
    return _DATASET


def data_config(**over):
    dc = dict(cams=None, Ncams=None, input_size=(384, 704), src_size=(720, 1280),
              resize=(-0.06, 0.11), rot=(-5.4, 5.4), flip=True, crop_h=(0.0, 0.0),
              resize_test=0.0, re_ratio=False,
              pitch_aug=(0.0, 0.0), yaw_aug=(0.0, 0.0), roll_aug=(0.0, 0.0),
              extri_x_aug=(0.0, 0.0), extri_y_aug=(0.0, 0.0), extri_z_aug=(0.0, 0.0))
    dc.update(over)
    return dc


def run_pipeline(index, dc, aug=None, heatmaps=True):
    """PrepareImageInputs_UDA -> LoadAnnotationsBEVDepth (no BDA) -> Load3DBoxesHeatmap."""
    from mmdet3d.datasets.pipelines.loading_UDA import PrepareImageInputs_UDA, Load3DBoxesHeatmap
    from mmdet3d.datasets.pipelines.loading import LoadAnnotationsBEVDepth
    ds = pccr_dataset()
    results = ds.get_data_info(index)
    ds.pre_pipeline(results)
    prep = PrepareImageInputs_UDA(data_config=dc, is_train=True, ego_cam=None)
    if aug is not None:
        prep.sample_augmentation = lambda H, W, flip=None, scale=None: aug
    results = prep(results)
    bda = dict(rot_lim=(0.0, 0.0), scale_lim=(1.0, 1.0), flip_dx_ratio=0.0, flip_dy_ratio=0.0)
    results = LoadAnnotationsBEVDepth(bda_aug_conf=bda, classes=CLASSES)(results)
    if heatmaps:
        kw = {}
        import inspect
        if 'point_cloud_range' in inspect.signature(Load3DBoxesHeatmap.__init__).parameters:
            kw['point_cloud_range'] = PC_RANGE
        results = Load3DBoxesHeatmap(classes=CLASSES, downsample_feature=8, **kw)(results)
    return results


def quat_mat(q):
    from pyquaternion import Quaternion
    return Quaternion(q).rotation_matrix


def reference_centres(results, view='real', ds=8):
    """Expected 2D target centres per camera: objects of the 7 classes inside the BEV range,
    in front of the camera (z > 1 m), whose augmented centre lies inside the feature map."""
    gt_boxes, gt_labels = results['ann_infos']
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 9)
    gt_labels = np.asarray(gt_labels)
    keep = (gt_labels >= 0) & (gt_labels < len(CLASSES))
    keep &= (np.abs(gt_boxes[:, 0]) <= 51.2) & (np.abs(gt_boxes[:, 1]) <= 51.2)
    imgs, rots, trans, intrins = results['img_inputs'][:4]
    H8, W8 = imgs.shape[-2] // ds, imgs.shape[-1] // ds
    out = []
    for i, cam in enumerate(results['cam_names']):
        if view == 'real':
            info = results['curr']['cams'][cam]
        else:
            info = results['bev_aug'][f'cam{i}']
        R = quat_mat(info['sensor2ego_rotation'])
        t = np.asarray(info['sensor2ego_translation'], dtype=np.float64)
        pc = (gt_boxes[:, :3] - t) @ R           # ego -> camera
        K = np.asarray(intrins[i], dtype=np.float64)
        uvz = pc @ K.T
        z = uvz[:, 2:3]
        uv = uvz[:, :2] / np.where(np.abs(z) < 1e-6, 1e-6, z)
        post = results['img_aug'][f'cam{i}']
        prot, ptran = np.asarray(post[1])[:2, :2], np.asarray(post[2])[:2]
        uv = (uv @ prot.T + ptran) / ds
        ok = keep & (pc[:, 2] > 1.0) & (uv[:, 0] >= 0) & (uv[:, 0] < W8) & (uv[:, 1] >= 0) & (uv[:, 1] < H8)
        out.append([(int(math.floor(u)), int(math.floor(v)), int(l)) for (u, v), l, k in zip(uv, gt_labels, ok) if k])
    return out, (H8, W8)


def pick_sample(min_ignored=3):
    """First R1 mini sample with at least `min_ignored` objects outside the 7 classes."""
    ds = pccr_dataset()
    for i in range(len(ds)):
        labels = np.asarray(ds.data_infos[i]['ann_infos'][1])
        if (labels < 0).sum() >= min_ignored:
            return i
    return 0


# ---------------------------------------------------------------- checks
def check_pdbev1():
    """IFV rendering must read bev[b, :, z, y, x] for the voxel (x, y, z)."""
    from mmdet3d.models.necks.view_transformer_pc import LSSViewTransformer_pc as VT
    Z, Y, X = 2, 3, 5                                   # non-square on purpose
    bev = torch.zeros(1, 1, Z, Y, X)
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                bev[0, 0, z, y, x] = 100 * z + 10 * y + x
    pts = [(x, y, z) for z in range(Z) for y in range(Y) for x in range(X)]
    coor = torch.tensor(pts, dtype=torch.long).view(1, 1, 1, 1, len(pts), 3)
    stub = SimpleNamespace(grid_size=torch.tensor([X, Y, 1]), BEV_aux_cfg=SimpleNamespace(height_num=Z))
    want = torch.tensor([100 * z + 10 * y + x for x, y, z in pts], dtype=torch.float32)
    try:
        got = VT.get_feature_Voxel2Frutums(stub, bev, coor).view(-1)
    except Exception as e:  # the transposed index runs past dim Y=3 for x >= 3
        return report('pdbev1', False, f'{type(e).__name__}: {e}')
    wrong = int((got != want).sum())
    return report('pdbev1', wrong == 0, f'{wrong}/{len(pts)} voxels read from the wrong cell')


def check_pdbev3():
    """The virtual camera used for rendering must sit where its targets are made (t + delta)."""
    delta = (1.0, 0.5, 0.3)
    dc = data_config(extri_x_aug=(delta[0], delta[0]), extri_y_aug=(delta[1], delta[1]),
                     extri_z_aug=(delta[2], delta[2]))
    r = run_pipeline(0, dc, aug=(0.55, (704, 396), (0, 12, 704, 396), False, 0.0), heatmaps=False)
    errs = []
    for i in range(len(r['cam_names'])):
        render_t = np.asarray(r['bev_aug']['tran_augs'][i], dtype=np.float64)
        target_t = np.asarray(r['bev_aug'][f'cam{i}']['sensor2ego_translation'], dtype=np.float64)
        errs.append(np.linalg.norm(render_t - target_t))
    return report('pdbev3', max(errs) < 1e-4,
                  f'render vs target camera position: max {max(errs):.3f} m over {len(errs)} cameras')


def _targets(index, aug):
    r = run_pipeline(index, data_config(), aug=aug)
    return r, r['heatmaps_2d'], r['heatmap_masks_2d'], r['ann_maps_2d']


def check_pdbev4():
    """Objects outside the 7 classes (label -1) must not be drawn into the last class channel."""
    idx = pick_sample()
    r, hm, mask, ann = _targets(idx, (0.55, (704, 396), (0, 12, 704, 396), False, 0.0))
    ref, _ = reference_centres(r)
    want = sum(1 for cam in ref for (_, _, l) in cam if l == len(CLASSES) - 1)
    got = int((hm[..., -1] == 1).sum())
    n_ign = int((np.asarray(r['ann_infos'][1]) < 0).sum())
    return report('pdbev4', got == want,
                  f'sample {idx} ({n_ign} objects with label -1): {got} peaks in the last class '
                  f'channel "{CLASSES[-1]}", expected {want}')


def check_pdbev5():
    """2D targets only for the 7 classes inside the BEV range (as the 3D GT after the filters)."""
    idx = pick_sample()
    r, hm, mask, ann = _targets(idx, (0.55, (704, 396), (0, 12, 704, 396), False, 0.0))
    ref, _ = reference_centres(r)
    want = sum(len(c) for c in ref)
    got = int((hm == 1).sum())
    return report('pdbev5', got == want, f'sample {idx}: {got} heatmap peaks, expected {want}')


def check_pdbev6():
    """Mask/regression windows must stay at the object (no writes wrapped to the far border)."""
    idx = pick_sample()
    # resize 0.66 and a bottom crop push the upper image rows above the feature map
    aug = (0.66, (844, 475), (60, 91, 764, 475), False, 0.0)
    r, hm, mask, ann = _targets(idx, aug)
    ref, (H8, W8) = reference_centres(r)
    bad = missing = 0
    for i, cam in enumerate(ref):
        want = torch.zeros(H8, W8, len(CLASSES), dtype=torch.bool)
        for u, v, l in cam:
            want[max(v - 1, 0):v + 2, max(u - 1, 0):u + 2, l] = True
        got = mask[i] > 0
        bad += int((got & ~want).sum())
        missing += int((want & ~got).sum())
    return report('pdbev6', bad == 0 and missing == 0,
                  f'sample {idx}: {bad} mask cells outside the objects\' windows, {missing} missing')


def check_pdbev8():
    """IFV z-bins must cover the whole z range for any height_num."""
    from mmcv import ConfigDict
    from mmdet3d.models.necks.view_transformer_pc import LSSViewTransformer_pc
    vt = LSSViewTransformer_pc(
        grid_config=dict(x=[-51.2, 51.2, 0.8], y=[-51.2, 51.2, 0.8], z=[-5, 3, 8], depth=[1.0, 100.0, 1.0]),
        input_size=(384, 704), in_channels=64, out_channels=8, downsample=16,
        BEV_Aux=ConfigDict(type='BEV_Aux', class_name=CLASSES, numC_input=32, num_layer=[1, 1, 1],
                           height_num=3, num_channels=[8, 8, 11]))
    covered = vt.revised_interval * 3
    return report('pdbev8', abs(covered - 8.0) < 1e-6, f'3 bins x {vt.revised_interval} m = {covered} m of 8 m')


def check_pdbev9():
    """Actual intrinsics must map camera points to the flipped image pixels."""
    r = run_pipeline(0, data_config(), aug=(0.55, (704, 396), (0, 12, 704, 396), True, 0.0), heatmaps=False)
    imgs, rots, trans, intrins, post_rots, post_trans, bda, act = r['img_inputs']
    err = 0.0
    for i in range(len(r['cam_names'])):
        X = np.array([1.0, 0.5, 10.0])
        p = np.asarray(intrins[i], dtype=np.float64) @ X
        p = p[:2] / p[2]
        p_aug = np.asarray(post_rots[i])[:2, :2] @ p + np.asarray(post_trans[i])[:2]
        q = np.asarray(act[i], dtype=np.float64)[:3, :3] @ X
        err = max(err, float(np.abs(q[:2] / q[2] - p_aug).max()))
    return report('pdbev9', err < 1e-3, f'flipped sample: actual-intrinsics pixel off by {err:.1f} px')


def check_pdbev10():
    """pitch_aug must tilt the camera (rotation about camera x), roll_aug spin it about the optical axis."""
    from mmdet3d.datasets.pipelines.loading_UDA import PrepareImageInputs_UDA
    cam = np.array([0.5, -0.5, 0.5, -0.5])               # front camera: x right, y down, z forward
    out = {}
    for key in ('pitch_aug', 'roll_aug'):
        dc = data_config(**{key: (0.1, 0.1)})
        q = PrepareImageInputs_UDA(data_config=dc).Euler_aug(cam)
        Rrel = quat_mat(cam).T @ quat_mat(q)             # perturbation in the camera frame
        w = np.array([Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0], Rrel[1, 0] - Rrel[0, 1]])
        out[key] = int(np.argmax(np.abs(w)))             # 0: camera x, 1: y, 2: z
    ok = out['pitch_aug'] == 0 and out['roll_aug'] == 2
    names = 'xyz'
    return report('pdbev10', ok, f'pitch_aug rotates about camera {names[out["pitch_aug"]]}, '
                                f'roll_aug about camera {names[out["roll_aug"]]} (want x and z)')


def _build_aux():
    from mmdet3d.models.necks.img_aux import Img_Aux, Img_Aux_Dy, BEV_Aux
    torch.manual_seed(0)
    c = len(CLASSES) * 7
    return dict(
        img_aux=Img_Aux(numC_input=32, class_name=CLASSES, upsample=[1, 2, 1], num_channels=[32, 32, c]),
        bev_img_aux=Img_Aux_Dy(numC_input=16, class_name=CLASSES, num_layer=[2, 1, 1], upsample=[2, 1, 1],
                               num_channels=[32, 32, c]),
        bev_aux=BEV_Aux(numC_input=32, class_name=CLASSES, num_layer=[2, 2, 2], height_num=4,
                        num_channels=[16, 16, 20]))


def _hook_bev_aux(m, seen):
    """Record the module outputs of BEV_Aux in call order; the last one is the head output."""
    mods = list(m.BEV_features) + [c for c in m.children() if isinstance(c, torch.nn.Conv2d)]
    for mod in mods:
        mod.register_forward_hook(lambda mm, i, o: seen.append(o))


def check_pdbev17():
    """Aux heatmaps must be able to go below 0.5 (the head output must not end in a ReLU)."""
    m = _build_aux()
    for v in m.values():
        v.eval()
    with torch.no_grad():
        h1 = m['img_aux'].get_heatmaps(torch.randn(2, 32, 12, 22))[0]
        h2 = m['bev_img_aux'].get_heatmaps(torch.randn(2, 16, 12, 22))[0]
        seen = []
        _hook_bev_aux(m['bev_aux'], seen)
        m['bev_aux'](torch.randn(2, 32, 16, 16))
        feats = seen[-1]
    hmin = [float(h1.min()), float(h2.min()), float(torch.sigmoid(feats[:, :4]).min())]
    ok = all(v < 0.5 for v in hmin)
    return report('pdbev17', ok, 'min of sigmoid(output): Img_Aux %.3f, Img_Aux_Dy %.3f, BEV_Aux height %.3f' % tuple(hmin))


def check_pdbev12():
    """IFV = sigmoid(F_height) * F_bev (paper eq. 3)."""
    m = _build_aux()['bev_aux'].eval()
    seen = []
    _hook_bev_aux(m, seen)
    with torch.no_grad():
        voxel, _ = m(torch.randn(2, 32, 16, 16))
    feats = seen[-1]
    want = torch.sigmoid(feats[:, :4]).unsqueeze(1) * feats[:, 4:].unsqueeze(2)
    err = float((voxel - want).abs().max())
    return report('pdbev12', err < 1e-5, f'max |IFV - sigmoid(F_height) * F_bev| = {err:.3g}')


def check_pdbev16():
    """Test-time crop must be deterministic and bottom-aligned like training (upstream formula)."""
    from mmdet3d.datasets.pipelines.loading_UDA import PrepareImageInputs_UDA
    prep = PrepareImageInputs_UDA(data_config=data_config(crop_h=(0.0, 0.0)), is_train=False)
    crops = {prep.sample_augmentation(720, 1280)[2] for _ in range(20)}
    newH = int(720 * 704 / 1280)
    want = int((1 - 0.0) * newH) - 384                    # training crop with crop_h (0, 0)
    ok = len(crops) == 1 and next(iter(crops))[1] == want
    return report('pdbev16', ok, f'{len(crops)} different test crops; top row {sorted(c[1] for c in crops)[:3]} (want {want})')


CHECKS = dict(pdbev1=check_pdbev1, pdbev3=check_pdbev3, pdbev4=check_pdbev4, pdbev5=check_pdbev5,
              pdbev6=check_pdbev6, pdbev8=check_pdbev8, pdbev9=check_pdbev9, pdbev10=check_pdbev10,
              pdbev12=check_pdbev12, pdbev16=check_pdbev16, pdbev17=check_pdbev17)


def main(argv):
    os.chdir(os.path.join(REPO, 'bevdepth'))
    names = argv or list(CHECKS)
    ok = True
    for n in names:
        try:
            ok &= bool(CHECKS[n]())
        except Exception as e:
            ok = False
            report(n, False, f'{type(e).__name__}: {e}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
