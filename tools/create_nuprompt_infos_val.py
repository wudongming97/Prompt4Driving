# ------------------------------------------------------------------------
# Copyright (c) 2024 Dongming Wu.
# ------------------------------------------------------------------------
# Modified from MUTR3D (https://github.com/a1600012888/MUTR3D)
# Copyright (c) 2022 Tianyuan Zhang
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import mmcv
import numpy as np
import os
from pyquaternion import Quaternion
from data_converter.nuscenes_prediction_tools import get_forecasting_annotations

import random
from mmdet3d.datasets import NuScenesDataset

random.seed(1314)
prompt_dir = '../data/nuscenes/nuprompt_v1.0'

#  remove the classes barrier, trafficcone and construction_vehicle
nus_categories = (
    'car', 'truck', 'bus', 'trailer',
    'motorcycle', 'bicycle', 'pedestrian',
    'construction_vehicle', 'traffic_cone', 'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')


def create_nuscenes_infos(root_path,
                          out_dir=None,
                          info_prefix=None,
                          version='v1.0-trainval',
                          max_sweeps=10,
                          forecasting=True,
                          forecasting_length=13):
    """Create info file of nuscene dataset.
    Given the raw data, generate its related info file in pkl format.
    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
        forecasting (bool): If prepare for forecasting data
        forecasting_length (int): Max frame number for forecasting.
            Default: 13 (6 seconds + current frame)
    """
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
        info_prefix = info_prefix + '-mini'
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))

    tracking_prompt_infos_val = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test=False, max_sweeps=max_sweeps,
        forecasting=forecasting, forecasting_length=forecasting_length)

    metadata = dict(version=version)
    data = dict(infos=tracking_prompt_infos_val, metadata=metadata)
    mmcv.dump(data, '../data/nuscenes/nuprompt_infos_val.pkl')
    print('Save NuPrompt validation file into {}'.format('../data/nuscenes/nuprompt_infos_val.pkl'))


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.
    Given the raw data, get the information of available scenes for
    further info generation.
    Args:
        nusc (class): Dataset class in the nuScenes dataset.
    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def _fill_trainval_infos(nusc,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10,
                         forecasting=True,
                         forecasting_length=13):
    """Generate the train/val infos from the raw data.
    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.
        forecasting (bool): If prepare for forecasting data
        forecasting_length (int): Max frame number for forecasting.
            Default: 13 (6 seconds + current frame)
    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    val_nusc_infos = []

    # get all the prompts
    nuprompt_dict = {}
    scene_list = os.listdir(prompt_dir)
    for scene in scene_list:
        prompt_files = os.listdir(os.path.join(prompt_dir, scene))
        nuprompt_dict[scene] = [os.path.join(prompt_dir, scene, prompt_file) for prompt_file in prompt_files]

    for scene in nusc.scene:

        if scene['token'] in train_scenes:
            continue
        print(f'processing {scene["name"]}')
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)

        if scene['token'] not in nuprompt_dict.keys():
            print(f'{scene["name"]} not in prompt dir')
            prompt_list = [None]
        else:
            prompt_list = nuprompt_dict[scene_token]

        if len(prompt_list) > 2:
            prompt_list = random.sample(prompt_list, 2)
        print(f'prompt_list: {prompt_list}')

        for prompt_filename in prompt_list:

            first_token = scene_rec['first_sample_token']
            last_token = scene_rec['last_sample_token']
            current_token = first_token

            frame_idx = 0
            while current_token != last_token:

                sample = nusc.get('sample', current_token)
                lidar_token = sample['data']['LIDAR_TOP']
                sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                cs_record = nusc.get('calibrated_sensor',
                                     sd_rec['calibrated_sensor_token'])
                pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
                lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

                info = {
                    'lidar_path': lidar_path,
                    'token': sample['token'],
                    'sweeps': [],
                    'cams': dict(),
                    'radars': dict(),
                    'lidar2ego_translation': cs_record['translation'],
                    'lidar2ego_rotation': cs_record['rotation'],
                    'ego2global_translation': pose_record['translation'],
                    'ego2global_rotation': pose_record['rotation'],
                    'timestamp': sample['timestamp'],
                    'scene_token': sample['scene_token'],
                    'frame_idx': frame_idx,
                    'prompt_filename': prompt_filename
                }

                if sample['next'] == '':
                    frame_idx = 0
                else:
                    frame_idx += 1

                l2e_r = info['lidar2ego_rotation']
                l2e_t = info['lidar2ego_translation']
                e2g_r = info['ego2global_rotation']
                e2g_t = info['ego2global_translation']
                l2e_r_mat = Quaternion(l2e_r).rotation_matrix
                e2g_r_mat = Quaternion(e2g_r).rotation_matrix

                # obtain 6 image's information per frame
                camera_types = [
                    'CAM_FRONT',
                    'CAM_FRONT_RIGHT',
                    'CAM_FRONT_LEFT',
                    'CAM_BACK',
                    'CAM_BACK_LEFT',
                    'CAM_BACK_RIGHT',
                ]
                for cam in camera_types:
                    cam_token = sample['data'][cam]
                    cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
                    cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                                 e2g_t, e2g_r_mat, cam)
                    cam_info.update(cam_intrinsic=cam_intrinsic)
                    info['cams'].update({cam: cam_info})

                # obtain sweeps for a single key-frame
                sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                sweeps = []
                while len(sweeps) < max_sweeps:
                    if not sd_rec['prev'] == '':
                        sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                                  l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                        sweeps.append(sweep)
                        sd_rec = nusc.get('sample_data', sd_rec['prev'])
                    else:
                        break
                info['sweeps'] = sweeps
                # obtain annotation
                if not test:
                    annotations = [
                        nusc.get('sample_annotation', token)
                        for token in sample['anns']
                    ]
                    locs = np.array([b.center for b in boxes]).reshape(-1, 3)
                    dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
                    rots = np.array([b.orientation.yaw_pitch_roll[0]
                                     for b in boxes]).reshape(-1, 1)
                    velocity = np.array(
                        [nusc.box_velocity(token)[:2] for token in sample['anns']])
                    valid_flag = np.array(
                        [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                         for anno in annotations],
                        dtype=bool).reshape(-1)
                    # convert velo from global to lidar
                    for i in range(len(boxes)):
                        velo = np.array([*velocity[i], 0.0])
                        velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                            l2e_r_mat).T
                        velocity[i] = velo[:2]

                    names = [b.name for b in boxes]
                    for i in range(len(names)):
                        if names[i] in NuScenesDataset.NameMapping:
                            names[i] = NuScenesDataset.NameMapping[names[i]]
                    names = np.array(names)
                    # update valid now
                    name_in_track = [_a in nus_categories for _a in names]
                    name_in_track = np.array(name_in_track)
                    valid_flag = np.logical_and(valid_flag, name_in_track)

                    # add instance_ids
                    instance_inds = [nusc.getind('instance', ann['instance_token']) for ann in annotations]
                    # we need to convert rot to SECOND format.
                    gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
                    assert len(gt_boxes) == len(
                        annotations), f'{len(gt_boxes)}, {len(annotations)}'
                    info['gt_boxes'] = gt_boxes
                    info['gt_names'] = names
                    info['gt_velocity'] = velocity.reshape(-1, 2)
                    info['num_lidar_pts'] = np.array(
                        [a['num_lidar_pts'] for a in annotations])
                    info['num_radar_pts'] = np.array(
                        [a['num_radar_pts'] for a in annotations])
                    info['valid_flag'] = valid_flag
                    info['instance_inds'] = instance_inds

                    if forecasting:
                        fboxes, fannotations, fmasks, ftypes = get_forecasting_annotations(nusc, annotations,
                                                                                           forecasting_length)
                        locs = [np.array([b.center for b in boxes]).reshape(-1, 3) for boxes in fboxes]
                        tokens = [np.array([b.token for b in boxes]) for boxes in fboxes]
                        info['forecasting_locs'] = np.array(locs)
                        info['forecasting_tokens'] = np.array(tokens)
                        info['forecasting_masks'] = np.array(fmasks)
                        info['forecasting_types'] = np.array(ftypes)

                current_ann = nusc.get('sample', current_token)

                next_token = current_ann['next']
                current_token = next_token

                if scene['token'] in val_scenes:
                    val_nusc_infos.append(info)

    return val_nusc_infos


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.
    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.
    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


if __name__ == '__main__':

    # generate .pkl for train, and val
    create_nuscenes_infos('../data/nuscenes/')

    # generate .pkl for test set
    # create_nuscenes_infos('data/nuscenes/', 'track_test', version='v1.0-test')