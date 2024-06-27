# ------------------------------------------------------------------------
# Copyright (c) 2024 Dongming Wu.
# ------------------------------------------------------------------------

import mmcv
import numpy as np
import pyquaternion
import tempfile
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp
from mmdet.datasets import DATASETS
from mmdet3d.core import show_result
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from pyquaternion import Quaternion
from .nuscenes_forecasting_bbox import NuScenesForecastingBox

import os
import random


@DATASETS.register_module()
class NuPromptDataset(NuScenesDataset):
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    AttrMapping = {
        'cycle.with_rider': 0,
        'cycle.without_rider': 1,
        'pedestrian.moving': 2,
        'pedestrian.standing': 3,
        'pedestrian.sitting_lying_down': 4,
        'vehicle.moving': 5,
        'vehicle.parked': 6,
        'vehicle.stopped': 7,
    }
    AttrMapping_rev = [
        'cycle.with_rider',
        'cycle.without_rider',
        'pedestrian.moving',
        'pedestrian.standing',
        'pedestrian.sitting_lying_down',
        'vehicle.moving',
        'vehicle.parked',
        'vehicle.stopped',
    ]
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    CLASSES = ('car', 'truck', 'bus', 'trailer',
               'motorcycle', 'bicycle', 'pedestrian',
               'construction_vehicle', 'traffic_cone', 'barrier')
    TRACKING_CLASSES = ['car', 'truck', 'bus', 'trailer',
                        'motorcycle', 'bicycle', 'pedestrian']

    def __init__(self,
                 pipeline_multiframe=None,
                 num_frames_per_sample=2,
                 forecasting=False,
                 ratio=1,
                 nuprompt_root='../data/nuscenes/nuprompt_v1.0',
                 *args, **kwargs,
                 ):
        self.num_frames_per_sample = num_frames_per_sample
        self.pipeline_multiframe = pipeline_multiframe
        self.forecasting = forecasting
        self.ratio = ratio  # divide the samples by a certain ratio, useful for quicker ablations
        if self.pipeline_multiframe is not None:
            self.pipeline_multiframe = Compose(self.pipeline_multiframe)

        # read prompt directory
        self.nuprompt_root = nuprompt_root
        self.nuprompt_dict = {}
        scene_list = os.listdir(self.nuprompt_root)
        for scene in scene_list:
            prompt_list = os.listdir(os.path.join(self.nuprompt_root, scene))
            self.nuprompt_dict[scene] = [os.path.join(self.nuprompt_root, scene, prompt) for prompt in prompt_list]
        self.instance_token_to_id_map = mmcv.load('./data/nuscenes/instance_token_to_id_map.pkl')

        self.filter_gt_from_nuprompt = True
        self.filter_pred_of_nuprompt = 0.3

        # self.nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes/', verbose=True)

        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        # if testing, we use unsorted data because the ann_file has been sorted
        # if training, we use sorted data to ensure the order of data_infos
        if self.test_mode:
            data_infos = list(data['infos'])
        else:
            data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        data_infos = data_infos[::self.load_interval]
        return data_infos

    def __len__(self):
        if not self.test_mode:
            return len(self.data_infos) // self.ratio
        else:
            return len(self.data_infos)
        # return len(self.data_infos)

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
            scene_token=info['scene_token']
        )

        # ego movement represented by lidar2global
        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix
        l2e, e2g = np.eye(4), np.eye(4)
        l2e[:3, :3], l2e[:3, 3] = l2e_r_mat, l2e_t
        e2g[:3, :3], e2g[:3, 3] = e2g_r_mat, e2g_t

        l2g_r_mat = l2e_r_mat @ e2g_r_mat  # [3, 3]
        l2g_t = l2e_t @ e2g_r_mat + e2g_t  # [1, 3]
        l2g = e2g @ l2e

        # points @ R.T + T
        input_dict.update(
            dict(
                l2g_r_mat=l2g_r_mat.astype(np.float32),
                l2g_t=l2g_t.astype(np.float32),
                l2g=l2g.astype(np.float32)))

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                                  'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)
                intrinsics.append(viewpad)
                extrinsics.append(lidar2cam_rt)  # the transpose of extrinsic matrix

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                ))

        if self.test_mode:
            input_dict.update(dict(prompt=info['prompt_filename']))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        instance_inds = np.array(info['instance_inds'], dtype=np.int)[mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # before [x, y, z, w, l, h, rot, vx, vy]
        # after [x, y, z-h/2, w, l, h, rot, vx, vy]
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            instance_inds=instance_inds)

        # load forecasting information
        if self.forecasting:
            anns_results['gt_forecasting_locs'] = info['forecasting_locs'][mask]
            anns_results['gt_forecasting_masks'] = info['forecasting_masks'][mask]
            anns_results['gt_forecasting_types'] = info['forecasting_types'][mask]
        return anns_results

    def pre_pipeline(self, results):
        """Initialization before data preparation.
        Args:
            results (dict): Dict before data preprocessing.
                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def generate_track_data_indexes(self, index):
        """Choose the track indexes that are within the same sequence
        """
        index_list = [i for i in range(index - self.num_frames_per_sample + 1, index + 1)]
        scene_tokens = [self.data_infos[i]['scene_token'] for i in index_list]
        tgt_scene_token, earliest_index = scene_tokens[-1], index_list[-1]
        for i in range(self.num_frames_per_sample)[::-1]:
            if scene_tokens[i] == tgt_scene_token:
                earliest_index = index_list[i]
            elif self.test_mode:
                index_list = index_list[i + 1:]
                break
            elif (not self.test_mode):
                index_list[i] = earliest_index
        return index_list

    def prepare_train_data(self, index):
        """ Sample multiple frames and check if they come from the same scene
        """
        data_queue = list()

        # index_list = range(index - self.num_frames_per_sample + 1, index + 1)
        index_list = self.generate_track_data_indexes(index)
        index_list = index_list[::-1]

        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or ~(example['gt_labels_3d'] != -1).any()):
            return None
        scene_token = input_dict['scene_token']
        data_queue.append(example)

        for i in index_list[1:]:
            data_info_i = self.get_data_info(i)
            if data_info_i is None:
                return None
            # use scene token to check if the sampled frames come from the same sequence
            if data_info_i['scene_token'] != scene_token:
                return None
            self.pre_pipeline(data_info_i)
            example = self.pipeline(data_info_i)

            if self.filter_empty_gt and \
                    (example is None or
                     ~(example['gt_labels_3d'] != -1).any()):
                return None
            data_queue.append(example)

        # return to the normal frame order
        data_queue = data_queue[::-1]

        # create data
        sample_data = dict()
        for key in data_queue[-1].keys():
            sample_data[key] = list()
        for d in data_queue:
            for k, v in d.items():
                sample_data[k].append(v)

        # after reading multi-frame data, we read the prompt
        # if there is prompt, then we record whether the object is referred
        is_ref_list = []
        if scene_token in list(self.nuprompt_dict.keys()):
            prompt_path = random.choice(self.nuprompt_dict[scene_token])
            prompt_dict = mmcv.load(prompt_path)
            # prompt_dict format
            # 'scene_token': 'scene_token',
            # 'prompt': ['prompt1', 'prompt2', ...],
            # 'original_prompt': ['original prompt'],
            # 'prompt_frame_object': {'frame_token1': ['object1', 'object2', ...], ...}}
            prompt = random.choice(prompt_dict['prompt'])
            frame_token_object_token = prompt_dict['frame_token_object_token']
            for frame_i, frame_token in enumerate(sample_data['sample_idx']):
                # if the frame is in the prompt-matching frames, then record whether the object is referred
                # else all objects are not referred
                is_ref = []
                if frame_token in frame_token_object_token:
                    object_tokens = frame_token_object_token[frame_token]
                    object_ids = [int(self.instance_token_to_id_map[object_token]) for object_token in object_tokens]
                    for frame_i_instance_id in sample_data['instance_inds'][frame_i]:
                        if int(frame_i_instance_id) in object_ids:
                            is_ref.append(1)
                        else:
                            is_ref.append(0)
                else:
                    is_ref = list(np.zeros(len(sample_data['instance_inds'][frame_i])))
                is_ref_list.append(is_ref)
            sample_data['prompt'] = [prompt] * self.num_frames_per_sample
            sample_data['prompt_object_ids'] = is_ref_list
        # if this scene has no prompt, then all objects are referred
        else:
            sample_data['prompt'] = ['all objects'] * self.num_frames_per_sample
            for frame_i, frame_token in enumerate(sample_data['sample_idx']):
                is_ref = list(np.ones(len(sample_data['instance_inds'][frame_i])))
                is_ref_list.append(is_ref)
            sample_data['prompt_object_ids'] = is_ref_list

        # multiframe processing
        data = self.pipeline_multiframe(sample_data)
        return data

    def prepare_test_data(self, index):
        data_queue = list()

        index_list = self.generate_track_data_indexes(index)
        index_list = index_list[::-1]

        input_dict = self.get_data_info(index_list[0])
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        scene_token = input_dict['scene_token']
        data_queue.append(example)

        for i in index_list[1:]:
            data_info_i = self.get_data_info(i)
            if data_info_i is None:
                return None
            # use scene token to check if the sampled frames come from the same sequence
            if data_info_i['scene_token'] != scene_token:
                return None
            self.pre_pipeline(data_info_i)
            example = self.pipeline(data_info_i)
            data_queue.append(example)

        # return to the normal frame order
        data_queue = data_queue[::-1]

        # create data
        sample_data = dict()
        for key in data_queue[-1].keys():
            sample_data[key] = list()
        for d in data_queue:
            for k, v in d.items():
                sample_data[k].append(v)

        # read prompts
        prompt_file = sample_data['prompt'][0]
        if prompt_file is None:
            sample_data['prompt'] = ['all objects'] * self.num_frames_per_sample
        else:
            prompt_dict = mmcv.load(prompt_file.replace('../data/nuscenes/nuprompt_v1.0', './data/nuscenes/nuprompt_v1.0'))
            # only choose the first prompt if having multiple ones
            prompt = prompt_dict['prompt'][0]
            sample_data['prompt'] = [prompt] * self.num_frames_per_sample

        # multiframe processing
        data = self.pipeline_multiframe(sample_data)
        return data

    def evaluate_prompt_tracking(self,
                                 results,
                                 metric='bbox',
                                 logger=None,
                                 jsonfile_prefix=None,
                                 result_names=['pts_bbox'],
                                 show=False,
                                 out_dir=None,
                                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        # Here, we filter the prompt-referred objects in terms of prompt_scores
        result_files, tmp_dir = self.format_prompt_tracking_results(results, jsonfile_prefix)

        # Here, we evaluate prompt_tracking results
        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_tracking_prompt_single(result_files[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_tracking_prompt_single(result_files)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict

    def format_prompt_tracking_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_prompt_tracking_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_prompt_tracking_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def _format_prompt_tracking_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.TRACKING_CLASSES
        det_class_names = self.CLASSES
        self.prompt_class = []

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det)
            sample_token = self.data_infos[sample_id]['token']
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                             det_class_names,
                                             self.eval_detection_configs,
                                             self.eval_version,
                                             tracking=True)

            # choose the prompt
            prompt_path = self.data_infos[sample_id]['prompt_filename']
            if prompt_path == None:
                prompt_path = 'all-objects'
                prompt = 'all objects'
            else:
                prompt_path = prompt_path.replace('../data/nuscenes/nuprompt_v1.0', './data/nuscenes/nuprompt_v1.0')
                prompt = mmcv.load(prompt_path)['prompt'][0]
            scene_token = self.data_infos[sample_id]['scene_token']

            for i, box in enumerate(boxes):
                # We define a prompt class via a format of "scene_token + * + prompt"
                # The attr is randomly chosen
                name = scene_token + '*' + prompt
                if name not in self.prompt_class:
                    self.prompt_class.append(name)
                attr = 'vehicle.moving'

                if det['prompt_pred'][i] > self.filter_pred_of_nuprompt:
                    nusc_anno = dict(
                        sample_token=sample_token + '*' + prompt_path, # this info will be used in gt preparation
                        translation=box.center.tolist(),
                        size=box.wlh.tolist(),
                        rotation=box.orientation.elements.tolist(),
                        velocity=box.velocity[:2].tolist(),
                        tracking_name=name, # this info will be used for calculating tracking metrics by prompt
                        attribute_name=attr,
                        tracking_score=box.score,
                        tracking_id=box.token, )
                    if box.forecasting is not None:
                        traj = box.forecasting.tolist()
                        traj = [(traj_ + det['boxes_3d'].center[i][:2].numpy()).tolist() for traj_ in traj]
                        nusc_anno['forecasting'] = traj

                    annos.append(nusc_anno)

            nusc_annos[sample_token + '*' + prompt_path] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_prompt_tracking.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_tracking_prompt_single(self,
                                         result_path,
                                         logger=None,
                                         metric='bbox',
                                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        # from nuscenes import NuScenes
        output_dir = osp.join(*osp.split(result_path)[:-1])

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        # from nuscenes.eval.tracking.evaluate import TrackingEval
        detail = dict()

        from projects.tracking_plugin.datasets.utils.nuprompt_eval import PromptTrackingEval
        from nuscenes.eval.common.config import config_factory as track_configs

        cfg = track_configs("tracking_nips_2019")
        # The original TrackingEval is calculated with the class_names of nuScenes
        # We need to change multiple classes to one prompt class
        cfg.class_names = self.prompt_class
        nusc_eval = PromptTrackingEval(
            config=cfg,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True,
            nusc_version=self.version,
            nusc_dataroot=self.data_root,
            filter_gt_from_nuprompt=self.filter_gt_from_nuprompt
        )
        metrics = nusc_eval.main()
        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        print(metrics)
        metric_prefix = f'{result_name}_NuScenes'
        keys = ['amota', 'amotp', 'recall', 'motar',
                'gt', 'mota', 'motp', 'mt', 'ml', 'faf',
                'tp', 'fp', 'fn', 'ids', 'frag', 'tid', 'lgd']
        for key in keys:
            detail['{}/{}'.format(metric_prefix, key)] = metrics[key]

        return detail

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            inds = result['scores_3d'] > 0.1
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)


def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

        tracking (bool): if convert for tracking evaluation

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    if 'track_scores' in detection.keys() and detection['track_scores'] is not None:
        scores = detection['track_scores'].numpy()
    labels = detection['labels_3d'].numpy()

    if 'forecasting' in detection.keys() and detection['forecasting'] is not None:
        forecasting = detection['forecasting'].numpy()
    else:
        forecasting = [None for _ in range(len(box3d))]

    if 'track_ids' in detection.keys() and detection['track_ids'] is not None:
        track_ids = detection['track_ids']
    else:
        track_ids = [None for _ in range(len(box3d))]

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesForecastingBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
            forecasting=forecasting[i],
            token=str(track_ids[i]))
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019',
                             tracking=False):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        # filter out the classes not for tracking
        if tracking and classes[box.label] not in NuPromptDataset.TRACKING_CLASSES:
            continue
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list


def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix

def convert_egopose_to_matrix_numpy(rotation, translation):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix
