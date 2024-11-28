# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from __future__ import absolute_import

import numpy as np

from boxmot.motion.cmc import get_cmc_method
from boxmot.trackers.strongsort.sort import iou_matching, linear_assignment
from boxmot.trackers.strongsort.sort.track import Track
from boxmot.utils.matching import chi2inv95


class Tracker:
    """
    This is the multi-target tracker.
    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    tracks : List[Track]
        The list of active tracks at the current time step.
    """

    GATING_THRESHOLD = np.sqrt(chi2inv95[4])

    def __init__(
        self,
        metric,
        max_iou_dist=0.9,
        max_age=30,
        n_init=3,
        _lambda=0,
        ema_alpha=0.9,
        mc_lambda=0.995,
    ):
        self.metric = metric
        self.max_iou_dist = max_iou_dist
        self.max_age = max_age
        self.n_init = n_init
        self._lambda = _lambda
        self.ema_alpha = ema_alpha
        self.mc_lambda = mc_lambda

        self.tracks = []
        self._next_id = 1
        self.cmc = get_cmc_method('ecc')()

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:  # 对每个轨迹处理
            track.predict()  # 刷新卡尔曼滤波预测位置

    def increment_ages(self):
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    def update(self, detections):  # 关联轨迹与det
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)  # 进行级联匹配

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])  # 更新匹配上的轨迹
        for track_idx in unmatched_tracks:  # 没匹配上的轨迹
            self.tracks[track_idx].mark_missed()  # 设置轨迹为丢失
        for detection_idx in unmatched_detections:  # 没匹配上的det
            self._initiate_track(detections[detection_idx])  # 将没匹配上的det初始化
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric. 更新距离指标
        active_targets = [t.id for t in self.tracks if t.is_confirmed()]  # 提取匹配上的指标
        features, targets = [], []
        for track in self.tracks:  # 遍历每个轨迹
            if not track.is_confirmed():
                continue
            features += track.features  # 更新特征
            targets += [track.id for _ in track.features]
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets
        )  # 使用新数据更新轨迹

    def _match(self, detections):  # 进行匹配
        def gated_metric(tracks, dets, track_indices, detection_indices):  # 门控
            features = np.array([dets[i].feat for i in detection_indices])  # 读取det特征
            targets = np.array([tracks[i].id for i in track_indices])  # 读取轨迹id
            cost_matrix = self.metric.distance(features, targets)  # 计算相似度矩阵
            print("reid_dist:")
            print(cost_matrix)
            cost_matrix = linear_assignment.gate_cost_matrix(  # 对相似度矩阵进行门控过滤
                cost_matrix,  # 待过滤的相似度矩阵
                tracks,
                dets,
                track_indices,  # 轨迹
                detection_indices,  # det
                self.mc_lambda,  # 0.995
            )  #
            print("after gate_cost_matrix:")
            print(cost_matrix)
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks. 轨迹分类
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]  # 匹配过的轨迹
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]  # 新轨迹

        # Associate confirmed tracks using appearance features. 用reid匹配关联上的轨迹
        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
            gated_metric,   # 门控
            self.metric.matching_threshold,  # 0.4
            self.max_age,  # 30
            self.tracks,  # 所有轨迹
            detections,  # 所有det
            confirmed_tracks,  # 匹配关联上的轨迹
        )  # 使用reid关联activate的轨道

        # Associate remaining tracks together with unconfirmed tracks using IOU.  # 用iou关联丢失 及新轨迹 和reid没匹配上的轨迹
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]  # 丢失及新轨迹 加上reid没匹配上的轨迹中上一帧匹配上的轨迹
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1
        ]  # reid没匹配上的轨迹中的丢失轨迹

        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
            iou_matching.iou_cost,  #
            self.max_iou_dist,  # 0.7
            self.tracks,  # 所有轨迹
            detections,  # 所有det
            iou_track_candidates,  # 丢失及新轨迹 加上reid没匹配上的轨迹中上一帧匹配上的轨迹
            unmatched_detections,  # reid没匹配上的det
        )  # iou关联

        matches = matches_a + matches_b  # 合并reid和iou关联上的
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))  # 没关联上的轨迹
        return matches, unmatched_tracks, unmatched_detections  #

    def _initiate_track(self, detection):  # 创建轨迹
        self.tracks.append(
            Track(
                detection,
                self._next_id,
                self.n_init,
                self.max_age,
                self.ema_alpha,
            )
        )
        self._next_id += 1
