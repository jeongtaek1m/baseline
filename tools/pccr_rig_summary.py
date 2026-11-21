#!/usr/bin/env python3
"""Summarize the 14 PCCR camera rigs from the benchmark's own rig definitions.

Reads third_party/pccr/pccr/configs/rigs/<RIG>.json (CARLA convention: location x forward,
y right, z up in metres; rotation = [pitch, yaw, roll] in degrees) and prints a markdown
table plus, for the controlled variants, what changed with respect to R1.

Usage: python3 tools/pccr_rig_summary.py [--rigs-dir DIR]
"""
import argparse
import json
import math
import os

RIGS = ['R1', 'R1-c6', 'R1-c10', 'R1-f', 'R1-r', 'R1-t',
        'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9']
IMG_W = 1280  # all PCCR cameras render 1280x720 (pccr/core/scene_runner.py)


def focal_px(fov_deg, width=IMG_W):
    return width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))


def load(rigs_dir, rig):
    with open(os.path.join(rigs_dir, f'{rig}.json')) as f:
        return json.load(f)


def summarize(cfg):
    cams = cfg['cameras']
    fovs = sorted({c['fov'] for c in cams})
    zs = [c['location'][2] for c in cams]
    pitches = sorted({c['rotation'][0] for c in cams})
    lidar = cfg.get('lidar', {}).get('location', [float('nan')] * 3)
    return dict(vehicle=cfg.get('vehicle_type', '?').replace('vehicle.', ''),
                ncams=len(cams), fovs=fovs, zmin=min(zs), zmax=max(zs),
                pitches=pitches, lidar=lidar)


def diff_vs_ref(ref, cfg):
    ref_c = {c['id']: c for c in ref['cameras']}
    cur_c = {c['id']: c for c in cfg['cameras']}
    notes = []
    added = sorted(set(cur_c) - set(ref_c))
    removed = sorted(set(ref_c) - set(cur_c))
    if added:
        notes.append('added ' + ', '.join(a.replace('CAM_', '') for a in added))
    if removed:
        notes.append('removed ' + ', '.join(r.replace('CAM_', '') for r in removed))
    for cid in sorted(set(ref_c) & set(cur_c)):
        a, b = ref_c[cid], cur_c[cid]
        changes = []
        if a['fov'] != b['fov']:
            changes.append(f"fov {a['fov']}->{b['fov']}")
        dyaw = b['rotation'][1] - a['rotation'][1]
        dpitch = b['rotation'][0] - a['rotation'][0]
        if abs(dyaw) > 1e-6:
            changes.append(f'yaw {dyaw:+.1f}')
        if abs(dpitch) > 1e-6:
            changes.append(f'pitch {dpitch:+.1f}')
        dloc = [b['location'][i] - a['location'][i] for i in range(3)]
        if max(abs(d) for d in dloc) > 1e-6:
            changes.append('loc ' + ','.join(f'{d:+.2f}' for d in dloc))
        if changes:
            notes.append(f"{cid.replace('CAM_', '')}: " + ' '.join(changes))
    return notes


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--rigs-dir', default=os.path.join(here, '..', 'third_party', 'pccr', 'pccr', 'configs', 'rigs'))
    args = ap.parse_args()

    ref = load(args.rigs_dir, 'R1')
    print('| Rig | Vehicle | #Cams | FOV (deg) -> f @1280px | Cam height (m) | Cam pitch (deg) | LiDAR xyz (m) |')
    print('|---|---|---|---|---|---|---|')
    for rig in RIGS:
        s = summarize(load(args.rigs_dir, rig))
        fov = ', '.join(f'{v} -> {focal_px(v):.0f}' for v in s['fovs'])
        lid = ', '.join(f'{v:.2f}' for v in s['lidar'])
        pit = ', '.join(f'{p:g}' for p in s['pitches'])
        print(f"| {rig} | {s['vehicle']} | {s['ncams']} | {fov} | {s['zmin']:.2f}-{s['zmax']:.2f} | {pit} | {lid} |")
    print()
    print('Controlled variants vs R1:')
    for rig in ['R1-c6', 'R1-c10', 'R1-f', 'R1-r', 'R1-t']:
        notes = diff_vs_ref(ref, load(args.rigs_dir, rig))
        print(f'- {rig}: ' + ('; '.join(notes) if notes else 'identical camera set'))


if __name__ == '__main__':
    main()
