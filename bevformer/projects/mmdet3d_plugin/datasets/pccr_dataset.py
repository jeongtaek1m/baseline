# Gen2E: PCCR (Plentiful CARLA Camera Rigs, ECCV 2026) dataset for single-frame BEVFormer.
import copy
from os import path as osp

import mmcv
import numpy as np
import pyquaternion
from mmdet.datasets import DATASETS
from nuscenes.utils.data_classes import Box as NuScenesBox

from .nuscenes_dataset import CustomNuScenesDataset
from .pccr_eval import PCCREvalMixin


def lidar_to_ref_boxes(boxes, lidar2ego_rotation, lidar2ego_translation, ref2ego_translation):
    """Move PCCR boxes (mmdet3d >= 1.0 layout, LIDAR_TOP frame) into the reference frame.

    The reference frame has the ego axes and its origin at ``ref2ego_translation`` in the ego
    frame, so ``p_ref = p_ego - ref2ego_translation``. LIDAR_TOP -> ego must be a rotation about
    z (in PCCR it is the identity for every rig) plus a translation.
    """
    boxes = np.array(boxes, copy=True)
    rot = pyquaternion.Quaternion(lidar2ego_rotation).rotation_matrix
    if abs(rot[2, 2] - 1.0) > 1e-6:
        raise ValueError('LIDAR_TOP -> ego is not a rotation about z: %s' % (lidar2ego_rotation,))
    boxes[:, :3] = boxes[:, :3] @ rot.T + np.asarray(lidar2ego_translation) \
        - np.asarray(ref2ego_translation)
    boxes[:, 6] = boxes[:, 6] + np.arctan2(rot[1, 0], rot[0, 0])  # yaw in the >= 1.0 layout
    return boxes


def pccr_to_mmdet3d_v017_boxes(boxes):
    """Convert PCCR ``gt_boxes`` to the mmdet3d 0.17.1 LiDAR box layout.

    PCCR infos store boxes in the mmdet3d >= 1.0 layout ``[x, y, z, l, w, h, yaw]`` (LIDAR_TOP
    frame, z = gravity center; third_party/pccr/pccr/lib/data_converter/nuscenes_converter.py:423
    writes ``dims[:, [1, 0, 2]], rots``). BEVFormer runs on mmdet3d 0.17.1, whose nuScenes
    converter writes ``dims`` = (w, l, h) and ``-rots - pi / 2``. Centers and heights are unchanged.
    """
    boxes = np.array(boxes, copy=True)
    out = boxes.copy()
    out[:, 3] = boxes[:, 4]  # w
    out[:, 4] = boxes[:, 3]  # l
    out[:, 6] = -boxes[:, 6] - np.pi / 2
    return out


@DATASETS.register_module()
class PCCRBEVFormerDataset(PCCREvalMixin, CustomNuScenesDataset):
    """PCCR for BEVFormer (mmdet3d 0.17.1), one frame per sample.

    Differences from CustomNuScenesDataset (projects/mmdet3d_plugin/datasets/nuscenes_dataset.py):

    * ``load_annotations``: rig-relative paths are resolved against ``data_root`` (as in PCCR's
      adapters) and ``gt_boxes`` are converted with :func:`pccr_to_mmdet3d_v017_boxes`.
    * reference frame (``ref2ego_translation``): ``None`` keeps upstream's frame, each rig's
      LIDAR_TOP. A 3-vector makes boxes, ``lidar2img`` and outputs use a vehicle-fixed frame
      instead: the ego axes with the origin at that point of the ego frame. ``lidar2img`` is then
      reference -> image, built from the cameras' sensor2ego calibration (the LiDAR calibration is
      not used). Gen2E uses ``[-0.46, 0, 1.86]``, R1's LIDAR_TOP mount, so the etron rigs (R1, R1-*,
      R2, R3, R7) see exactly the upstream LiDAR frame and the other vehicles get the same ground
      height (z = -1.86) instead of their own LiDAR height (decision 2026-11-22).
    * ``get_data_info``: PCCR infos have no ``next``, ``frame_idx`` or ``can_bus``. The CAN bus input
      is switched off (zeros; Gen2E decision 2026-11-22), so no ego pose or ego motion reaches the
      model. Cameras are all cameras of the rig in info order, as in CustomNuScenesDataset.
    * ``_format_bbox``: mmdet3d 0.17.1 LiDAR boxes -> nuScenes global boxes, without velocity.
    * evaluation: :class:`PCCREvalMixin`, PCCR's own nuScenes-devkit evaluation copied verbatim, the
      same code that scores the BEVDepth rows.
    """

    def __init__(self,
                 *args,
                 class_range=None,
                 discard_velocity_eval=True,
                 with_velocity=False,
                 ref2ego_translation=None,
                 **kwargs):
        # Set before super().__init__(), which calls load_annotations().
        self.ref2ego_translation = (None if ref2ego_translation is None else
                                    np.asarray(ref2ego_translation, dtype=np.float64))
        # As PCCR's adapter (third_party/pccr/models/BEVDet/pccr_dataset.py:89-102).
        if with_velocity:
            raise NotImplementedError(
                'PCCRBEVFormerDataset is trained without velocity; _format_bbox does not '
                'transform velocities to the global frame.')
        self.class_range = copy.deepcopy(self.DEFAULT_CLASS_RANGE)
        if class_range is not None:
            self.class_range.update(class_range)
        self.discard_velocity_eval = discard_velocity_eval
        super().__init__(*args, with_velocity=with_velocity, **kwargs)
        self.eval_detection_configs = self._build_eval_detection_configs()

    def load_annotations(self, ann_file):
        """As mmdet3d's NuScenesDataset.load_annotations, plus the two PCCR fixes."""
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        for info in data_infos:
            self._normalize_info_paths(info)
            boxes = info['gt_boxes']
            if self.ref2ego_translation is not None:
                boxes = lidar_to_ref_boxes(boxes, info['lidar2ego_rotation'],
                                           info['lidar2ego_translation'],
                                           self.ref2ego_translation)
            info['gt_boxes'] = pccr_to_mmdet3d_v017_boxes(boxes)
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def get_data_info(self, index):
        """CustomNuScenesDataset.get_data_info for PCCR infos (changes marked "Gen2E")."""
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            # Gen2E: PCCR infos have 'prev_token' but no 'next' or 'frame_idx'. These are
            # metadata only; the single-frame model does not use them.
            prev_idx=info.get('prev_token', ''),
            next_idx='',
            scene_token=info['scene_token'],
            # Gen2E: CAN bus off. Upstream copies info['can_bus'] and writes the ego pose into it
            # below; the model adds can_bus_mlp(can_bus) to every BEV query.
            can_bus=np.zeros(18),
            frame_idx=index,
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                sensor2ref_r, sensor2ref_t = self._sensor2ref(cam_info)  # Gen2E
                lidar2cam_r = np.linalg.inv(sensor2ref_r)
                lidar2cam_t = sensor2ref_t @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        # Gen2E: upstream writes the ego pose and yaw into can_bus here (CAN bus off: removed).
        return input_dict

    def _sensor2ref(self, cam_info):
        """Camera -> reference frame (Gen2E). Upstream (LIDAR_TOP frame) uses sensor2lidar."""
        if self.ref2ego_translation is None:
            return cam_info['sensor2lidar_rotation'], cam_info['sensor2lidar_translation']
        rot = pyquaternion.Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
        trans = np.asarray(cam_info['sensor2ego_translation']) - self.ref2ego_translation
        return rot, trans

    def _ref2ego(self, info):
        """Reference frame -> ego, as (Quaternion, translation)."""
        if self.ref2ego_translation is None:
            return (pyquaternion.Quaternion(info['lidar2ego_rotation']),
                    np.array(info['lidar2ego_translation']))
        return pyquaternion.Quaternion(), self.ref2ego_translation.copy()

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Write BEVFormer outputs as a nuScenes submission (results_nusc.json).

        Outputs are mmdet3d 0.17.1 LiDAR boxes in the reference frame (LIDAR_TOP or the
        vehicle-fixed frame, see the class docstring). The conversion is the one of mmdet3d 0.17.1
        (output_to_nusc_box: size = dims (w, l, h), yaw -> -yaw - pi/2; lidar_nusc_box_to_global:
        reference -> ego -> global with the info's calibration and ego pose). Unlike
        lidar_nusc_box_to_global there is no class-range pre-filter: the devkit
        applies the same class ranges, exactly as for PCCR's adapter and our BEVDepth rows. The
        submission format follows PCCR's adapter (velocity 0, PCCR default attributes).
        """
        nusc_annos = {}
        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            info = self.data_infos[sample_id]
            sample_token = info['token']
            boxes = det['boxes_3d']
            box_tensor = boxes.tensor.numpy()
            centers = boxes.gravity_center.numpy()
            wlh = boxes.dims.numpy()
            yaws = -boxes.yaw.numpy() - np.pi / 2
            scores = det['scores_3d'].numpy()
            labels = det['labels_3d'].numpy()

            ref2ego_rot, ref2ego_trans = self._ref2ego(info)
            ego2global_rot = pyquaternion.Quaternion(info['ego2global_rotation'])
            ego2global_trans = np.array(info['ego2global_translation'])

            annos = []
            for i in range(len(box_tensor)):
                name = self.CLASSES[labels[i]]
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=yaws[i])
                nusc_box = NuScenesBox(centers[i], wlh[i], quat)
                nusc_box.rotate(ref2ego_rot)
                nusc_box.translate(ref2ego_trans)
                nusc_box.rotate(ego2global_rot)
                nusc_box.translate(ego2global_trans)
                box_vel, attr = self._get_box_velocity_and_attr(name, box_tensor[i])
                annos.append(
                    dict(
                        sample_token=sample_token,
                        translation=nusc_box.center.tolist(),
                        size=nusc_box.wlh.tolist(),
                        rotation=nusc_box.orientation.elements.tolist(),
                        velocity=box_vel[:2],
                        detection_name=name,
                        detection_score=float(scores[i]),
                        attribute_name=attr,
                    ))
            nusc_annos.setdefault(sample_token, []).extend(annos)

        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }
        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path
