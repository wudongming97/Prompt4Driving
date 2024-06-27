# ------------------------------------------------------------------------
# Copyright (c) 2024 Dongming Wu.
# ------------------------------------------------------------------------

from nuscenes import NuScenes
from nuscenes.eval.tracking.data_classes import TrackingMetrics, TrackingMetricDataList, TrackingConfig, TrackingBox, \
    TrackingMetricData

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.detection.utils import category_to_detection_name

import mmcv
import json
import tqdm
import numpy as np
from bisect import bisect
from pyquaternion import Quaternion
from collections import defaultdict
from typing import List, Dict, DefaultDict, Tuple
from nuscenes.eval.common.data_classes import EvalBox


class TrackingBox(EvalBox):
    """ Data class used during tracking evaluation. Can be a prediction or ground truth."""

    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 ego_translation: Tuple[float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
                 num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
                 tracking_id: str = '',  # Instance id of this object.
                 tracking_name: str = '',  # The class name used in the tracking challenge.
                 tracking_score: float = -1.0):  # Does not apply to GT.

        super().__init__(sample_token, translation, size, rotation, velocity, ego_translation, num_pts)

        assert tracking_name is not None, 'Error: tracking_name cannot be empty!'
        # assert tracking_name in TRACKING_NAMES, 'Error: Unknown tracking_name %s' % tracking_name

        assert type(tracking_score) == float, 'Error: tracking_score must be a float!'
        assert not np.any(np.isnan(tracking_score)), 'Error: tracking_score may not be NaN!'

        # Assign.
        self.tracking_id = tracking_id
        self.tracking_name = tracking_name
        self.tracking_score = tracking_score

    def __eq__(self, other):
        return (self.sample_token == other.sample_token and
                self.translation == other.translation and
                self.size == other.size and
                self.rotation == other.rotation and
                self.velocity == other.velocity and
                self.ego_translation == other.ego_translation and
                self.num_pts == other.num_pts and
                self.tracking_id == other.tracking_id and
                self.tracking_name == other.tracking_name and
                self.tracking_score == other.tracking_score)

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'ego_translation': self.ego_translation,
            'num_pts': self.num_pts,
            'tracking_id': self.tracking_id,
            'tracking_name': self.tracking_name,
            'tracking_score': self.tracking_score
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(sample_token=content['sample_token'],
                   translation=tuple(content['translation']),
                   size=tuple(content['size']),
                   rotation=tuple(content['rotation']),
                   velocity=tuple(content['velocity']),
                   ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                   else tuple(content['ego_translation']),
                   num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                   tracking_id=content['tracking_id'],
                   tracking_name=content['tracking_name'],
                   tracking_score=-1.0 if 'tracking_score' not in content else float(content['tracking_score']))

def add_center_dist_prompt(nusc: NuScenes, eval_boxes: EvalBoxes):
    """
    Adds the cylindrical (xy) center distance from ego vehicle to each box.
    :param nusc: The NuScenes instance.
    :param eval_boxes: A set of boxes, either GT or predictions.
    :return: eval_boxes augmented with center distances.
    """
    for sample_token in eval_boxes.sample_tokens:
        sample_rec = nusc.get('sample', sample_token.split('*')[0])
        sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        for box in eval_boxes[sample_token]:
            # Both boxes and ego pose are given in global coord system, so distance can be calculated directly.
            # Note that the z component of the ego pose is 0.
            ego_translation = (box.translation[0] - pose_record['translation'][0],
                               box.translation[1] - pose_record['translation'][1],
                               box.translation[2] - pose_record['translation'][2])
            if isinstance(box, DetectionBox) or isinstance(box, TrackingBox):
                box.ego_translation = ego_translation
            else:
                raise NotImplementedError

    return eval_boxes


def filter_eval_boxes_prompt(nusc: NuScenes,
                             eval_boxes: EvalBoxes,
                             max_dist: Dict[str, float],
                             verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param max_dist: Maps the detection name to the eval distance threshold for that class.
    :param verbose: Whether to print to stdout.
    """
    # Retrieve box type for detectipn/tracking boxes.
    class_field = _get_box_class_field(eval_boxes)

    # Accumulators for number of filtered boxes.
    total, dist_filter, point_filter, bike_rack_filter = 0, 0, 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):

        # Filter on distance first.
        total += len(eval_boxes[sample_token])
        # eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if
        #                                   box.ego_dist < max_dist[box.__getattribute__(class_field)]]
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token]]
        dist_filter += len(eval_boxes[sample_token])

        # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if not box.num_pts == 0]
        point_filter += len(eval_boxes[sample_token])

        # Perform bike-rack filtering.
        sample_anns = nusc.get('sample', sample_token.split('*')[0])['anns']
        bikerack_recs = [nusc.get('sample_annotation', ann) for ann in sample_anns if
                         nusc.get('sample_annotation', ann)['category_name'] == 'static_object.bicycle_rack']
        bikerack_boxes = [Box(rec['translation'], rec['size'], Quaternion(rec['rotation'])) for rec in bikerack_recs]
        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            if box.__getattribute__(class_field) in ['bicycle', 'motorcycle']:
                in_a_bikerack = False
                for bikerack_box in bikerack_boxes:
                    if np.sum(points_in_box(bikerack_box, np.expand_dims(np.array(box.translation), axis=1))) > 0:
                        in_a_bikerack = True
                if not in_a_bikerack:
                    filtered_boxes.append(box)
            else:
                filtered_boxes.append(box)

        eval_boxes.boxes[sample_token] = filtered_boxes
        bike_rack_filter += len(eval_boxes.boxes[sample_token])

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After distance based filtering: %d" % dist_filter)
        print("=> After LIDAR and RADAR points based filtering: %d" % point_filter)
        print("=> After bike rack filtering: %d" % bike_rack_filter)

    return eval_boxes


def _get_box_class_field(eval_boxes: EvalBoxes) -> str:
    """
    Retrieve the name of the class field in the boxes.
    This parses through all boxes until it finds a valid box.
    If there are no valid boxes, this function throws an exception.
    :param eval_boxes: The EvalBoxes used for evaluation.
    :return: The name of the class field in the boxes, e.g. detection_name or tracking_name.
    """
    assert len(eval_boxes.boxes) > 0
    box = None
    for val in eval_boxes.boxes.values():
        if len(val) > 0:
            box = val[0]
            break
    if isinstance(box, DetectionBox):
        class_field = 'detection_name'
    elif isinstance(box, TrackingBox):
        class_field = 'tracking_name'
    else:
        raise Exception('Error: Invalid box type: %s' % box)

    return class_field


def create_tracks_prompt(all_boxes: EvalBoxes, nusc: NuScenes, eval_split: str, gt: bool) \
        -> Dict[str, Dict[int, List[TrackingBox]]]:
    """
    Returns all tracks for all scenes. Samples within a track are sorted in chronological order.
    This can be applied either to GT or predictions.
    :param all_boxes: Holds all GT or predicted boxes.
    :param nusc: The NuScenes instance to load the sample information from.
    :param eval_split: The evaluation split for which we create tracks.
    :param gt: Whether we are creating tracks for GT or predictions
    :return: The tracks.
    """
    # Only keep samples from this split.
    splits = create_splits_scenes()
    scene_tokens = set()
    for sample_token in all_boxes.sample_tokens:

        #####
        prompt_path = sample_token.split('*')[1]
        if prompt_path == 'all-objects':
            prompt = 'all objects'
        else:
            prompt = mmcv.load(prompt_path)['prompt'][0]
        #####

        scene_token = nusc.get('sample', sample_token.split('*')[0])['scene_token']
        scene = nusc.get('scene', scene_token)
        if scene['name'] in splits[eval_split]:
            scene_tokens.add(scene_token + '*' + prompt)

    # Tracks are stored as dict {scene_token: {timestamp: List[TrackingBox]}}.
    tracks = defaultdict(lambda: defaultdict(list))

    # Init all scenes and timestamps to guarantee completeness.
    for scene_token in scene_tokens:
        # Init all timestamps in this scene.
        scene = nusc.get('scene', scene_token.split('*')[0])
        cur_sample_token = scene['first_sample_token']
        while True:
            # Initialize array for current timestamp.
            cur_sample = nusc.get('sample', cur_sample_token)
            tracks[scene_token][cur_sample['timestamp']] = []

            # Abort after the last sample.
            if cur_sample_token == scene['last_sample_token']:
                break

            # Move to next sample.
            cur_sample_token = cur_sample['next']

    # Group annotations wrt scene and timestamp.
    for sample_token in all_boxes.sample_tokens:
        #####
        prompt_path = sample_token.split('*')[1]
        if prompt_path == 'all-objects':
            prompt = 'all objects'
        else:
            prompt = mmcv.load(prompt_path)['prompt'][0]
        #####
        sample_record = nusc.get('sample', sample_token.split('*')[0])
        scene_token = sample_record['scene_token']
        tracks[scene_token + '*' + prompt][sample_record['timestamp']] = all_boxes.boxes[sample_token]

    # Replace box scores with track score (average box score). This only affects the compute_thresholds method and
    # should be done before interpolation to avoid diluting the original scores with interpolated boxes.
    if not gt:
        for scene_id, scene_tracks in tracks.items():
            # For each track_id, collect the scores.
            track_id_scores = defaultdict(list)
            for timestamp, boxes in scene_tracks.items():
                for box in boxes:
                    track_id_scores[box.tracking_id].append(box.tracking_score)

            # Compute average scores for each track.
            track_id_avg_scores = {}
            for tracking_id, scores in track_id_scores.items():
                track_id_avg_scores[tracking_id] = np.mean(scores)

            # Apply average score to each box.
            for timestamp, boxes in scene_tracks.items():
                for box in boxes:
                    box.tracking_score = track_id_avg_scores[box.tracking_id]

    # Interpolate GT and predicted tracks.
    for scene_token in tracks.keys():
        tracks[scene_token] = interpolate_tracks(tracks[scene_token])

        if not gt:
            # Make sure predictions are sorted in in time. (Always true for GT).
            tracks[scene_token] = defaultdict(list, sorted(tracks[scene_token].items(), key=lambda kv: kv[0]))

    return tracks


def interpolate_tracks(tracks_by_timestamp: DefaultDict[int, List[TrackingBox]]) -> DefaultDict[int, List[TrackingBox]]:
    """
    Interpolate the tracks to fill in holes, especially since GT boxes with 0 lidar points are removed.
    This interpolation does not take into account visibility. It interpolates despite occlusion.
    :param tracks_by_timestamp: The tracks.
    :return: The interpolated tracks.
    """
    # Group tracks by id.
    tracks_by_id = defaultdict(list)
    track_timestamps_by_id = defaultdict(list)
    for timestamp, tracking_boxes in tracks_by_timestamp.items():
        for tracking_box in tracking_boxes:
            tracks_by_id[tracking_box.tracking_id].append(tracking_box)
            track_timestamps_by_id[tracking_box.tracking_id].append(timestamp)

    # Interpolate missing timestamps for each track.
    timestamps = tracks_by_timestamp.keys()
    interpolate_count = 0
    for timestamp in timestamps:
        for tracking_id, track in tracks_by_id.items():
            if track_timestamps_by_id[tracking_id][0] <= timestamp <= track_timestamps_by_id[tracking_id][-1] and \
                    timestamp not in track_timestamps_by_id[tracking_id]:

                # Find the closest boxes before and after this timestamp.
                right_ind = bisect(track_timestamps_by_id[tracking_id], timestamp)
                left_ind = right_ind - 1
                right_timestamp = track_timestamps_by_id[tracking_id][right_ind]
                left_timestamp = track_timestamps_by_id[tracking_id][left_ind]
                right_tracking_box = tracks_by_id[tracking_id][right_ind]
                left_tracking_box = tracks_by_id[tracking_id][left_ind]
                right_ratio = float(right_timestamp - timestamp) / (right_timestamp - left_timestamp)

                # Interpolate.
                tracking_box = interpolate_tracking_boxes(left_tracking_box, right_tracking_box, right_ratio)
                interpolate_count += 1
                tracks_by_timestamp[timestamp].append(tracking_box)

    return tracks_by_timestamp


def interpolate_tracking_boxes(left_box: TrackingBox, right_box: TrackingBox, right_ratio: float) -> TrackingBox:
    """
    Linearly interpolate box parameters between two boxes.
    :param left_box: A Trackingbox.
    :param right_box: Another TrackingBox
    :param right_ratio: Weight given to the right box.
    :return: The interpolated TrackingBox.
    """
    def interp_list(left, right, rratio):
        return tuple(
            (1.0 - rratio) * np.array(left, dtype=float)
            + rratio * np.array(right, dtype=float)
        )

    def interp_float(left, right, rratio):
        return (1.0 - rratio) * float(left) + rratio * float(right)

    # Interpolate quaternion.
    rotation = Quaternion.slerp(
        q0=Quaternion(left_box.rotation),
        q1=Quaternion(right_box.rotation),
        amount=right_ratio
    ).elements

    # Score will remain -1 for GT.
    tracking_score = interp_float(left_box.tracking_score, right_box.tracking_score, right_ratio)

    return TrackingBox(sample_token=right_box.sample_token,
                       translation=interp_list(left_box.translation, right_box.translation, right_ratio),
                       size=interp_list(left_box.size, right_box.size, right_ratio),
                       rotation=rotation,
                       velocity=interp_list(left_box.velocity, right_box.velocity, right_ratio),
                       ego_translation=interp_list(left_box.ego_translation, right_box.ego_translation,
                                                   right_ratio),  # May be inaccurate.
                       tracking_id=right_box.tracking_id,
                       tracking_name=right_box.tracking_name,
                       tracking_score=tracking_score)


def load_gt_prompt(nusc: NuScenes, pred_path: str, eval_split: str, box_cls, verbose: bool = False,
                   filter_gt_from_nuprompt: bool = False) -> EvalBoxes:
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """
    # Init.
    if box_cls == DetectionBox:
        attribute_map = {a['token']: a['name'] for a in nusc.attribute}

    if verbose:
        print('Loading annotations for {} split from nuScenes version: {}'.format(eval_split, nusc.version))
    # Read out all sample_tokens in DB.
    sample_tokens_all = [s['token'] for s in nusc.sample]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    # Only keep samples from this split.
    splits = create_splits_scenes()

    # Check compatibility of split with nusc_version.
    version = nusc.version
    if eval_split in {'train', 'val', 'train_detect', 'train_track'}:
        assert version.endswith('trainval'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split == 'test':
        assert version.endswith('test'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    else:
        raise ValueError('Error: Requested split {} which this function cannot map to the correct NuScenes version.'
                         .format(eval_split))

    if eval_split == 'test':
        # Check that you aren't trying to cheat :).
        assert len(nusc.sample_annotation) > 0, \
            'Error: You are trying to evaluate on the test set but you do not have the annotations!'

    sample_tokens = []
    for sample_token in sample_tokens_all:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene_record = nusc.get('scene', scene_token)
        if scene_record['name'] in splits[eval_split]:
            sample_tokens.append(sample_token)

    # Load from file and check that the format is correct.
    with open(pred_path) as f:
        pred_data = json.load(f)

    sample_tokens_from_pred = [k for k in pred_data['results'].keys()]

    all_annotations = EvalBoxes()

    # Load annotations and filter predictions and annotations.
    tracking_id_set = set()
    for sample_token in tqdm.tqdm(sample_tokens_from_pred, leave=verbose):

        if filter_gt_from_nuprompt:
            prompt_path = sample_token.split('*')[1]
            # 1. if prompt_path is 'all-objects', load all gt annotations.
            if prompt_path == 'all-objects':
                keep_all_objects = True
                prompt = 'all objects'
            else:
                keep_all_objects = False
                with open(prompt_path, 'r') as f:
                    prompt_dict = json.load(f)
                prompt = prompt_dict['prompt'][0]
                frame_token_object_token = prompt_dict['frame_token_object_token']
                # 2. if sample_token is in frame_token_object_token, load object tokens for gt annotations.
                if sample_token.split('*')[0] in frame_token_object_token:
                    object_tokens = frame_token_object_token[sample_token.split('*')[0]]
                # 3. if sample_token is not in frame_token_object_token, generate empty gt annotations.
                else:
                    sample_boxes = []
                    all_annotations.add_boxes(sample_token, sample_boxes)
                    continue

        sample = nusc.get('sample', sample_token.split('*')[0])
        sample_annotation_tokens = sample['anns']

        sample_boxes = []
        for sample_annotation_token in sample_annotation_tokens:

            sample_annotation = nusc.get('sample_annotation', sample_annotation_token)
            # Considering above 2: skip gt annotation collection
            # if keep_all_objects is False and instance_token is not in object_tokens.
            if filter_gt_from_nuprompt and keep_all_objects is False and sample_annotation['instance_token'] not in object_tokens:
                continue

            if box_cls == DetectionBox:
                # Get label name in detection task and filter unused labels.
                detection_name = category_to_detection_name(sample_annotation['category_name'])
                if detection_name is None:
                    continue

                # Get attribute_name.
                attr_tokens = sample_annotation['attribute_tokens']
                attr_count = len(attr_tokens)
                if attr_count == 0:
                    attribute_name = ''
                elif attr_count == 1:
                    attribute_name = attribute_map[attr_tokens[0]]
                else:
                    raise Exception('Error: GT annotations must not have more than one attribute!')

                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        detection_name=detection_name,
                        detection_score=-1.0,  # GT samples do not have a score.
                        attribute_name=attribute_name
                    )
                )
            elif box_cls == TrackingBox:
                # Use nuScenes token as tracking id.
                tracking_id = sample_annotation['instance_token']
                tracking_id_set.add(tracking_id)

                # Get label name in detection task and filter unused labels.
                # Import locally to avoid errors when motmetrics package is not installed.
                from nuscenes.eval.tracking.utils import category_to_tracking_name
                # tracking_name = category_to_tracking_name(sample_annotation['category_name'])
                tracking_name = sample['scene_token'] + '*' + prompt
                if tracking_name is None:
                    continue

                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        tracking_id=tracking_id,
                        tracking_name=tracking_name,
                        tracking_score=-1.0  # GT samples do not have a score.
                    )
                )
            else:
                raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

        all_annotations.add_boxes(sample_token, sample_boxes)

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

    return all_annotations