# ------------------------------------------------------------------------
# Copyright (c) 2024 Dongming Wu.
# ------------------------------------------------------------------------
import os
import mmcv
import argparse
import numpy as np

from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from projects.tracking_plugin.visualization import NuscenesTrackingBox as TrackingBox
from projects.mmdet3d_plugin.datasets.pipelines.loading import LoadMultiViewImageFromNori


COLOR_MAP = {
    'red': np.array([191, 4, 54]) / 256,
    'light_blue': np.array([4, 157, 217]) / 256,
    'black': np.array([0, 0, 0]) / 256,
    'gray': np.array([140, 140, 136]) / 256,
    'purple': np.array([224, 133, 250]) / 256,
    # 'dark_green': np.array([32, 64, 40]) / 256,
    'green': np.array([77, 115, 67]) / 256,
    'brown': np.array([164, 103, 80]) / 256,
    # 'light_green': np.array([135, 206, 191]) / 256,
    'orange': np.array([229, 116, 57]) / 256,
}
COLOR_KEYS = list(COLOR_MAP.keys())


cams = ['CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_FRONT_LEFT']

import numpy as np
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from PIL import Image
from matplotlib import rcParams


def get_predicted_data(sample_data_token: str,
                       box_vis_level: BoxVisibility = BoxVisibility.ANY,
                       selected_anntokens=None,
                       use_flat_vehicle_coordinates: bool = False,
                       pred_anns=None
                       ):
    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    # if selected_anntokens is not None:
    #    boxes = list(map(nusc.get_box, selected_anntokens))
    # else:
    #    boxes = nusc.get_boxes(sample_data_token)
    boxes = pred_anns
    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue
        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def render_sample_data(
        sample_token: str,
        with_anns: bool = True,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        axes_limit: float = 40,
        ax=None,
        nsweeps: int = 1,
        out_path: str = None,
        underlay_map: bool = True,
        use_flat_vehicle_coordinates: bool = True,
        show_lidarseg: bool = False,
        show_lidarseg_legend: bool = False,
        filter_lidarseg_labels=None,
        lidarseg_preds_bin_path: str = None,
        verbose: bool = False,
        show_panoptic: bool = False,
        pred_data=None,
        load_from_nori=None
      ) -> None:

    sample = nusc.get('sample', sample_token.split('*')[0])
    cams = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
    ]
    for ind, cam in enumerate(cams):
        sample_data_token = sample['data'][cam]
        sensor_modality = 'camera'

        if sensor_modality in ['lidar', 'radar']:
            assert False
        elif sensor_modality == 'camera':
            # Load boxes and image.
            boxes = [TrackingBox(
                         'predicted',
                         record['translation'], record['size'], Quaternion(record['rotation']),
                         tracking_id=record['tracking_id'].split('-')[-1]) for record in
                     pred_data['results'][sample_token] if record['tracking_score'] > 0.4]

            data_path, boxes_pred, camera_intrinsic = get_predicted_data(sample_data_token,
                                                                         box_vis_level=box_vis_level, pred_anns=boxes)
            data_path = data_path.replace('../data/nuscenes', '/data/datasets/nuScenes')
            data_path = dict(img_filename=[data_path])
            data = load_from_nori(data_path)
            data = data['img'][0]
            data[:, :, [0, 1, 2]] = data[:, :, [2, 1, 0]]
            data = Image.fromarray(data.astype('uint8')).convert('RGB')
            # Show image.
            _, ax = plt.subplots(1, 1, figsize=(24, 18))
            ax.imshow(data)

            # Show boxes.
            if with_anns:
                for box in boxes_pred:
                    # c = np.array(get_color(box.name)) / 255.0
                    track_id = int(box.tracking_id)
                    color = COLOR_MAP[COLOR_KEYS[track_id % len(COLOR_KEYS)]]
                    box.render(ax, view=camera_intrinsic, normalize=True, \
                        colors=color, linestyle='solid', linewidth=4, text=False)

            # Limit visible range.
            ax.set_xlim(0, data.size[0])
            ax.set_ylim(data.size[1], 0)

            vis_path = os.path.join(out_path, cam + '.jpg')
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(vis_path, bbox_inches='tight', pad_inches=-0.1)
            plt.close()
            print('Saved to', vis_path)

        else:
            raise ValueError("Error: Unknown sensor modality!")


def parse_args():
    parser = argparse.ArgumentParser(description='Prompt Tracking Visualization')
    parser.add_argument('--result', help='results file', default='../work_dirs/f3_prompttrack/results_prompt_tracking.json')
    parser.add_argument('--data_infos_path', type=str, default='../data/nuscenes/nuprompt_infos_val.pkl')
    parser.add_argument('--scene_token', type=str, default='44c9089913db4d4ab839a2fcb35989ed')
    parser.add_argument('--prompt_filename', type=str, default='The-bus-is-of-the-white-color.json')
    parser.add_argument(
        '--show-dir', help='directory where visualize results will be saved', default='../work_dirs/f3_prompttrack/visualization')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    data_infos = mmcv.load(args.data_infos_path)['infos']

    nusc = NuScenes(version='v1.0-trainval', dataroot='../data/nuscenes/', verbose=True)

    predict = mmcv.load(args.result)
    predict_tokens = list(predict['results'].keys())

    load_from_nori = LoadMultiViewImageFromNori(to_float32=True,
                                                nori_lists=['s3://yjj/datasets/nuscenes_img_train.nori.list',
                                                            's3://yjj/datasets/nuscenes_img_val.nori.list',
                                                            's3://yjj/datasets/nuscenes_img_test.nori.list'],
                                                data_prefix='./data/nuscenes')

    pbar = tqdm(total=len(predict_tokens))
    for i, predict_token in enumerate(predict_tokens):
        # prepare the directory for visualization
        sample_info = data_infos[i]
        scene_token = sample_info['scene_token']
        if scene_token != args.scene_token:
            continue
        if args.prompt_filename != predict_token.split('/')[-1]:
            continue
        print('scene_token:', scene_token)
        print('prompt filename:', predict_token)

        out_path = os.path.join(args.show_dir, scene_token, predict_token.split('*')[1].split('/')[-1], predict_token.split('*')[0])
        os.makedirs(out_path, exist_ok=True)

        # render
        render_sample_data(predict_token, pred_data=predict, out_path=out_path, load_from_nori=load_from_nori)
        pbar.update(1)
    pbar.close()
