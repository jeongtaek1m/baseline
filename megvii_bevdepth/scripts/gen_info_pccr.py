# Gen2E: Megvii-format infos for PCCR (Plentiful CARLA Camera Rigs).
#
# Same per-sample record as scripts/gen_info.py (generate_info), with two PCCR changes:
#   * camera names come from the rig (4 to 10 cameras with rig-specific names) instead of the
#     six hard-coded nuScenes names, in the order of PCCR's own info pkl (the order every other
#     PCCR row uses);
#   * the samples of a split are exactly those of PCCR's info pkl <RIG>_infos_<split>.pkl
#     (nuscenes.utils.splits only knows nuScenes scene names), so the train/val/test samples are
#     the ones every other row of the table uses.
# Everything else (ego poses, calibrated sensors, sweeps, annotations with devkit velocity) is
# generate_info's code.
#
#   python scripts/gen_info_pccr.py --rig R1 --splits train val test
#   python scripts/gen_info_pccr.py --rig R5 --splits test
import argparse
import os

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'pccr')
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'pccr_infos')


def sample_info(nusc, cur_sample, cam_names, max_cam_sweeps=6, max_lidar_sweeps=10):
    """generate_info's per-sample body (scripts/gen_info.py) with `cam_names` as an argument."""
    info = dict()
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
        sweep_cam_info['calibrated_sensor'] = nusc.get(
            'calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_infos[cam_name] = sweep_cam_info
    for lidar_name in lidar_names:
        lidar_data = nusc.get('sample_data', cur_sample['data'][lidar_name])
        lidar_datas.append(lidar_data)
        sweep_lidar_info = dict()
        sweep_lidar_info['sample_token'] = lidar_data['sample_token']
        sweep_lidar_info['ego_pose'] = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        sweep_lidar_info['timestamp'] = lidar_data['timestamp']
        sweep_lidar_info['filename'] = lidar_data['filename']
        sweep_lidar_info['calibrated_sensor'] = nusc.get(
            'calibrated_sensor', lidar_data['calibrated_sensor_token'])
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
            sweep_cam_info['calibrated_sensor'] = nusc.get(
                'calibrated_sensor', cam_data['calibrated_sensor_token'])
            cam_sweeps[j][cam_names[k]] = sweep_cam_info
    for k, lidar_data in enumerate(lidar_datas):
        sweep_lidar_data = lidar_data
        for j in range(max_lidar_sweeps):
            if sweep_lidar_data['prev'] == '':
                break
            sweep_lidar_data = nusc.get('sample_data', sweep_lidar_data['prev'])
            sweep_lidar_info = dict()
            sweep_lidar_info['sample_token'] = sweep_lidar_data['sample_token']
            if sweep_lidar_info['sample_token'] != lidar_data['sample_token']:
                break
            sweep_lidar_info['ego_pose'] = nusc.get('ego_pose', sweep_lidar_data['ego_pose_token'])
            sweep_lidar_info['timestamp'] = sweep_lidar_data['timestamp']
            sweep_lidar_info['is_key_frame'] = sweep_lidar_data['is_key_frame']
            sweep_lidar_info['filename'] = sweep_lidar_data['filename']
            sweep_lidar_info['calibrated_sensor'] = nusc.get(
                'calibrated_sensor', cam_data['calibrated_sensor_token'])
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
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rig', required=True)
    ap.add_argument('--splits', nargs='+', default=['test'])
    ap.add_argument('--data-root', default=DEFAULT_ROOT)
    ap.add_argument('--out-dir', default=DEFAULT_OUT)
    args = ap.parse_args()
    root = os.path.join(os.path.abspath(args.data_root), args.rig)
    os.makedirs(args.out_dir, exist_ok=True)
    nusc_cache = {}
    for split in args.splits:
        pccr = mmcv.load(os.path.join(root, f'{args.rig}_infos_{split}.pkl'))
        version = pccr['metadata']['version']  # v1.0-trainval (train/val) or v1.0-test
        if version not in nusc_cache:
            nusc_cache[version] = NuScenes(version=version, dataroot=root, verbose=False)
        nusc = nusc_cache[version]
        infos = []
        for p in tqdm(pccr['infos'], desc=f'{args.rig} {split}'):
            cam_names = list(p['cams'].keys())
            infos.append(sample_info(nusc, nusc.get('sample', p['token']), cam_names))
        out = os.path.join(args.out_dir, f'{args.rig}_megvii_infos_{split}.pkl')
        mmcv.dump(infos, out)
        print(f'{args.rig} {split}: {len(infos)} samples, {len(infos[0]["cam_infos"])} cameras -> {out}')


if __name__ == '__main__':
    main()
