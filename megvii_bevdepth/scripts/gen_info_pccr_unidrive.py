# PCCR port: new file. Megvii-format infos for the UniDrive rows from the warped data of our BEVDet UniDrive rows.
"""UniDrive's warped PCCR data already exists (CoIn3D/pccr/make_unidrive_data.py: each rig's images warped into a
virtual rig, 800 x 450, BEVDet-format infos whose 'cams' are the virtual cameras):
  paper: /data1/giyong/coin3d_pccr_unidrive     (UniDrive's released virtual rig, 6 x 60 deg, roof centre, 1.6 m)
  r1:    /data1/giyong/coin3d_pccr_unidrive_r1  (R1's own 8 cameras as the virtual rig)
This script takes the Megvii infos of the raw rigs (scripts/gen_info_pccr.py: same tokens, LiDAR, annotations, ego
poses) and replaces only cam_infos by the virtual cameras: filename = the warped JPEG (absolute path), width / height,
calibrated_sensor = (sensor2ego rotation / translation, intrinsic) of the virtual camera, ego_pose = the sample's
(make_unidrive_data copies the first real camera's, = the LiDAR's in CARLA; asserted). The LiDAR stays the raw rig's
(relative path, data_root = the raw rig folder), so the official online depth GT projects every LiDAR point into the
virtual cameras.
    python scripts/gen_info_pccr_unidrive.py --src /data1/giyong/coin3d_pccr_unidrive --out data/pccr_megvii_unidrive
    python scripts/gen_info_pccr_unidrive.py --src /data1/giyong/coin3d_pccr_unidrive_r1 \
        --out data/pccr_megvii_unidrive_r1
"""
import argparse
import copy
import os
import pickle

import mmcv
import numpy as np
from PIL import Image

TEST_RIGS = ['R1', 'R1-c6', 'R1-c10', 'R1-f', 'R1-r', 'R1-t', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9']


def convert(raw_infos, src_infos):
    src = {i['token']: i for i in src_infos}
    assert set(src) == {i['sample_token'] for i in raw_infos}, 'token sets differ'
    out, n_missing = [], 0
    for raw in raw_infos:
        s = src[raw['sample_token']]
        le = raw['lidar_infos']['LIDAR_TOP']['ego_pose']
        cam_infos = {}
        for name, c in s['cams'].items():
            assert np.allclose(c['ego2global_rotation'], le['rotation'], atol=1e-9)
            assert np.allclose(c['ego2global_translation'], le['translation'], atol=1e-6)
            if not os.path.exists(c['data_path']):
                n_missing += 1
            cam_infos[name] = dict(
                sample_token=raw['sample_token'],
                ego_pose=dict(le),  # the sample's ego pose
                timestamp=int(c['timestamp']),
                is_key_frame=True,
                height=int(c['img_h']),
                width=int(c['img_w']),
                filename=c['data_path'],  # absolute: os.path.join(data_root, abs) == abs
                calibrated_sensor=dict(rotation=list(map(float, c['sensor2ego_rotation'])),
                                       translation=list(map(float, c['sensor2ego_translation'])),
                                       camera_intrinsic=np.asarray(c['cam_intrinsic'], dtype=np.float64).tolist()),
            )
        new = copy.copy(raw)
        new['cam_infos'] = cam_infos
        new['cam_sweeps'] = []
        out.append(new)
    return out, n_missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='make_unidrive_data.py output (R1_infos_train.pkl, test/*.pkl)')
    ap.add_argument('--raw', default='data/pccr_megvii', help='scripts/gen_info_pccr.py output')
    ap.add_argument('--out', required=True)
    ap.add_argument('--rigs', nargs='+', default=TEST_RIGS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs = [('R1', 'train', os.path.join(args.src, 'R1_infos_train.pkl'))]
    jobs += [(rig, 'test', os.path.join(args.src, 'test', f'{rig}_infos_test.pkl')) for rig in args.rigs]
    for rig, split, src_pkl in jobs:
        raw = mmcv.load(os.path.join(args.raw, f'{rig}_infos_{split}.pkl'))
        with open(src_pkl, 'rb') as f:
            src = pickle.load(f)['infos']
        infos, n_missing = convert(raw, src)
        c0 = next(iter(infos[0]['cam_infos'].values()))
        with Image.open(c0['filename']) as im:
            real = im.size
        assert real == (c0['width'], c0['height']), (real, c0['width'], c0['height'])
        out = os.path.join(args.out, f'{rig}_infos_{split}.pkl')
        mmcv.dump(infos, out)
        cams = list(infos[0]['cam_infos'].keys())
        K = np.array(c0['calibrated_sensor']['camera_intrinsic'])
        print(f'{rig} {split}: {len(infos)} samples, {len(cams)} virtual cams {cams}, image {real}, '
              f'fx {K[0, 0]:.2f}, missing images {n_missing} -> {out}', flush=True)


if __name__ == '__main__':
    main()
