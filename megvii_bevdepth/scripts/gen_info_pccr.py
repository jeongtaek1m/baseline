# PCCR port: new file. scripts/gen_info.py's generate_info for the PCCR benchmark (nuScenes-format tables per rig).
"""Megvii-format infos for PCCR.

    python scripts/gen_info_pccr.py --pccr-root /data4/pccr/data --out data/pccr_megvii

Writes <out>/R1_infos_train.pkl (R1, v1.0-trainval) and <out>/<rig>_infos_test.pkl (v1.0-test) for the 14 test rigs.
The samples are exactly the sample tokens of PCCR's own infos (<rig>/<rig>_infos_{train,test}.pkl), in their order,
so the splits match every other PCCR row. Everything else is official generate_info: per camera the sample_data
(filename relative to the rig folder, ego_pose, calibrated_sensor with the intrinsic), LIDAR_TOP likewise, empty
sweeps (PCCR has key frames only; the official prev-walk stops at the previous sample), and every sample_annotation
with nusc.box_velocity (NaN -> 0) in the global frame. The dataset turns them into the ego frame of the key sample.
Changes vs official (marked # PCCR port): the rig's own camera names, the token list, no scene-name split.
"""
import argparse
import os
import pickle

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

TEST_RIGS = ['R1', 'R1-c6', 'R1-c10', 'R1-f', 'R1-r', 'R1-t', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9']


def generate_info(nusc, sample_tokens, cam_names, max_cam_sweeps=6, max_lidar_sweeps=10):  # PCCR port: tokens, cams
    infos = list()
    for token in tqdm(sample_tokens):  # PCCR port: PCCR's own sample list instead of the scenes of a split
        cur_sample = nusc.get('sample', token)
        info = dict()
        sweep_cam_info = dict()
        cam_datas = list()
        lidar_datas = list()
        info['sample_token'] = cur_sample['token']
        info['timestamp'] = cur_sample['timestamp']
        info['scene_token'] = cur_sample['scene_token']
        lidar_names = ['LIDAR_TOP']
        cam_infos = dict()
        lidar_infos = dict()
        for cam_name in cam_names:
            cam_data = nusc.get('sample_data', cur_sample['data'][cam_name])
            cam_datas.append(cam_data)
            sweep_cam_info = dict()
            sweep_cam_info['sample_token'] = cam_data['sample_token']
            sweep_cam_info['ego_pose'] = nusc.get('ego_pose', cam_data['ego_pose_token'])
            sweep_cam_info['timestamp'] = cam_data['timestamp']
            sweep_cam_info['is_key_frame'] = cam_data['is_key_frame']
            sweep_cam_info['height'] = cam_data['height']
            sweep_cam_info['width'] = cam_data['width']
            sweep_cam_info['filename'] = cam_data['filename']
            sweep_cam_info['calibrated_sensor'] = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            cam_infos[cam_name] = sweep_cam_info
        for lidar_name in lidar_names:
            lidar_data = nusc.get('sample_data', cur_sample['data'][lidar_name])
            lidar_datas.append(lidar_data)
            sweep_lidar_info = dict()
            sweep_lidar_info['sample_token'] = lidar_data['sample_token']
            sweep_lidar_info['ego_pose'] = nusc.get('ego_pose', lidar_data['ego_pose_token'])
            sweep_lidar_info['timestamp'] = lidar_data['timestamp']
            sweep_lidar_info['filename'] = lidar_data['filename']
            sweep_lidar_info['calibrated_sensor'] = nusc.get('calibrated_sensor',
                                                             lidar_data['calibrated_sensor_token'])
            lidar_infos[lidar_name] = sweep_lidar_info

        lidar_sweeps = [dict() for _ in range(max_lidar_sweeps)]
        cam_sweeps = [dict() for _ in range(max_cam_sweeps)]
        info['cam_infos'] = cam_infos
        info['lidar_infos'] = lidar_infos
        for k, cam_data in enumerate(cam_datas):
            sweep_cam_data = cam_data
            for j in range(max_cam_sweeps):
                if sweep_cam_data['prev'] == '':
                    break
                else:
                    sweep_cam_data = nusc.get('sample_data', sweep_cam_data['prev'])
                    sweep_cam_info = dict()
                    sweep_cam_info['sample_token'] = sweep_cam_data['sample_token']
                    if sweep_cam_info['sample_token'] != cam_data['sample_token']:
                        break
                    sweep_cam_info['ego_pose'] = nusc.get('ego_pose', cam_data['ego_pose_token'])
                    sweep_cam_info['timestamp'] = sweep_cam_data['timestamp']
                    sweep_cam_info['is_key_frame'] = sweep_cam_data['is_key_frame']
                    sweep_cam_info['height'] = sweep_cam_data['height']
                    sweep_cam_info['width'] = sweep_cam_data['width']
                    sweep_cam_info['filename'] = sweep_cam_data['filename']
                    sweep_cam_info['calibrated_sensor'] = nusc.get('calibrated_sensor',
                                                                   cam_data['calibrated_sensor_token'])
                    cam_sweeps[j][cam_names[k]] = sweep_cam_info

        for k, lidar_data in enumerate(lidar_datas):
            sweep_lidar_data = lidar_data
            for j in range(max_lidar_sweeps):
                if sweep_lidar_data['prev'] == '':
                    break
                else:
                    sweep_lidar_data = nusc.get('sample_data', sweep_lidar_data['prev'])
                    sweep_lidar_info = dict()
                    sweep_lidar_info['sample_token'] = sweep_lidar_data['sample_token']
                    if sweep_lidar_info['sample_token'] != lidar_data['sample_token']:
                        break
                    sweep_lidar_info['ego_pose'] = nusc.get('ego_pose', sweep_lidar_data['ego_pose_token'])
                    sweep_lidar_info['timestamp'] = sweep_lidar_data['timestamp']
                    sweep_lidar_info['is_key_frame'] = sweep_lidar_data['is_key_frame']
                    sweep_lidar_info['filename'] = sweep_lidar_data['filename']
                    sweep_lidar_info['calibrated_sensor'] = nusc.get('calibrated_sensor',
                                                                     cam_data['calibrated_sensor_token'])
                    lidar_sweeps[j][lidar_names[k]] = sweep_lidar_info
        # Remove empty sweeps.
        for i, sweep in enumerate(cam_sweeps):
            if len(sweep.keys()) == 0:
                cam_sweeps = cam_sweeps[:i]
                break
        for i, sweep in enumerate(lidar_sweeps):
            if len(sweep.keys()) == 0:
                lidar_sweeps = lidar_sweeps[:i]
                break
        info['cam_sweeps'] = cam_sweeps
        info['lidar_sweeps'] = lidar_sweeps
        ann_infos = list()
        if 'anns' in cur_sample:
            for ann in cur_sample['anns']:
                ann_info = nusc.get('sample_annotation', ann)
                velocity = nusc.box_velocity(ann_info['token'])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info['velocity'] = velocity
                ann_infos.append(ann_info)
            info['ann_infos'] = ann_infos
        infos.append(info)
    return infos


def rig_dir(root, rig):  # PCCR port: <root>/<rig>/v1.0-* (Poseidon, NEO) or <root>/<rig>/<rig>/v1.0-* (shannon)
    d = os.path.join(root, rig)
    return d if os.path.isdir(os.path.join(d, 'v1.0-test')) else os.path.join(d, rig)


def check(infos, ref):  # PCCR port: the checks this port relies on
    assert [i['sample_token'] for i in infos] == [r['token'] for r in ref]
    n_bad_pose = 0
    for info in infos:
        le = info['lidar_infos']['LIDAR_TOP']['ego_pose']
        for c in info['cam_infos'].values():
            # one ego pose per sample (CARLA renders every sensor on one tick): the dataset's mean camera pose is the
            # sample's ego pose, the frame of PCCR's GT and of the devkit's ego distances
            if not (np.allclose(c['ego_pose']['rotation'], le['rotation'], atol=1e-9)
                    and np.allclose(c['ego_pose']['translation'], le['translation'], atol=1e-6)):
                n_bad_pose += 1
    return n_bad_pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pccr-root', default='/data4/pccr/data')
    ap.add_argument('--out', default='data/pccr_megvii')
    ap.add_argument('--rigs', nargs='+', default=TEST_RIGS)
    ap.add_argument('--skip-train', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs = [] if args.skip_train else [('R1', 'train', 'v1.0-trainval')]
    jobs += [(rig, 'test', 'v1.0-test') for rig in args.rigs]
    for rig, split, version in jobs:
        root = rig_dir(args.pccr_root, rig)
        with open(os.path.join(root, f'{rig}_infos_{split}.pkl'), 'rb') as f:
            ref = pickle.load(f)['infos']
        cams = list(ref[0]['cams'].keys())  # PCCR port: the rig's cameras in the order of PCCR's infos
        nusc = NuScenes(version=version, dataroot=root, verbose=False)
        infos = generate_info(nusc, [r['token'] for r in ref], cams)
        n_bad_pose = check(infos, ref)
        out = os.path.join(args.out, f'{rig}_infos_{split}.pkl')
        mmcv.dump(infos, out)
        sizes = sorted({(c['width'], c['height']) for i in infos for c in i['cam_infos'].values()})
        n_ann = sum(len(i['ann_infos']) for i in infos)
        n_pts = sum(a['num_lidar_pts'] + a['num_radar_pts'] > 0 for i in infos for a in i['ann_infos'])
        print(f'{rig} {split}: {len(infos)} samples ({len({i["scene_token"] for i in infos})} scenes), '
              f'{len(cams)} cams {cams}, image sizes {sizes}, {n_ann} annotations ({n_pts} with points), '
              f'cam/lidar ego-pose mismatches {n_bad_pose}, sweeps {sum(len(i["cam_sweeps"]) for i in infos)} '
              f'-> {out}', flush=True)


if __name__ == '__main__':
    main()
