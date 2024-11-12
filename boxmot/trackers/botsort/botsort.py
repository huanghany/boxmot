# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import torch
import numpy as np
from pathlib import Path

from boxmot.motion.kalman_filters.xywh_kf import KalmanFilterXYWH
from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.motion.cmc.sof import SOF
from boxmot.trackers.botsort.basetrack import BaseTrack, TrackState
from boxmot.utils.matching import (embedding_distance, fuse_score,
                                   iou_distance, linear_assignment)
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.botsort.botsort_utils import joint_stracks, sub_stracks, remove_duplicate_stracks 
from boxmot.trackers.botsort.botsort_track import STrack
from boxmot.motion.cmc import get_cmc_method



class BotSort(BaseTracker):
    """
    BoTSORT Tracker: A tracking algorithm that combines appearance and motion-based tracking.

    Args:
        reid_weights (str): Path to the model weights for ReID.
        device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
        half (bool): Use half-precision (fp16) for faster inference.
        per_class (bool, optional): Whether to perform per-class tracking.
        track_high_thresh (float, optional): Detection confidence threshold for first association.第一次关联的检测置信度
        track_low_thresh (float, optional): Detection confidence threshold for ignoring detections. 低于这个置信度的不追踪
        new_track_thresh (float, optional): Threshold for creating a new track.  建立追踪的阈值
        track_buffer (int, optional): Frames to keep a track alive after last detection. 追踪缓冲 丢了之后追踪器保留几帧
        match_thresh (float, optional): Matching threshold for data association.  数据关联的阈值
        proximity_thresh (float, optional): IoU threshold for first-round association.  第一次关联的iou阈值
        appearance_thresh (float, optional): Appearance embedding distance threshold for ReID.  reid相似度关联阈值
        cmc_method (str, optional): Method for correcting camera motion, e.g., "sof" (simple optical flow). 校正相机运动的算法选择
        frame_rate (int, optional): Video frame rate, used to scale the track buffer.  帧率
        fuse_first_associate (bool, optional): Fuse appearance and motion in the first association step. 第一次关联时是否融合运动和关联
        with_reid (bool, optional): Use ReID features for association. 是否使用reid模型
    """

    def __init__(
        self,
        reid_weights: Path,
        device: torch.device,
        half: bool,
        per_class: bool = False,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,  #
        new_track_thresh: float = 0.6,  #
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        cmc_method: str = "ecc",
        frame_rate=30,
        fuse_first_associate: bool = False,  #
        with_reid: bool = True,
    ):
        super().__init__(per_class=per_class)
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.per_class = per_class
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)  # 30
        self.max_time_lost = self.buffer_size  # 最大丢失时间 30
        self.kalman_filter = KalmanFilterXYWH()  #

        # ReID module
        self.proximity_thresh = proximity_thresh  # 第一次关联的iou阈值
        self.appearance_thresh = appearance_thresh  # 相似度关联阈值
        self.with_reid = with_reid  # 是否使用reid模型
        # self.with_reid = False
        if self.with_reid:   # 使用reid
            self.model = ReidAutoBackend(
                weights=reid_weights, device=device, half=half
            ).model

        self.cmc = get_cmc_method(cmc_method)()
        self.fuse_first_associate = fuse_first_associate

    @BaseTracker.on_first_frame_setup
    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:  # 更新追踪器
        self.check_inputs(dets, img)
        self.frame_count += 1

        activated_stracks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        # Preprocess detections
        dets, dets_first, embs_first, dets_second = self._split_detections(dets, embs)

        # Extract appearance features
        if self.with_reid and embs is None:
            features_high = self.model.get_features(dets_first[:, 0:4], img)  # 获取相似度
        else:
            features_high = embs_first if embs_first is not None else []

        # Create detections 创建追踪
        detections = self._create_detections(dets_first, features_high)

        # Separate unconfirmed and active tracks 将已经追踪上的和没追踪上的分开
        unconfirmed, active_tracks = self._separate_tracks()
        
        strack_pool = joint_stracks(active_tracks, self.lost_stracks)

        # First association 第一次关联
        matches_first, u_track_first, u_detection_first = self._first_association(dets, dets_first, active_tracks, unconfirmed, img, detections, activated_stracks, refind_stracks, strack_pool)

        # Second association 第二次关联
        matches_second, u_track_second, u_detection_second = self._second_association(dets_second, activated_stracks, lost_stracks, refind_stracks, u_track_first, strack_pool)

        # Handle unconfirmed tracks 处理没有关联的
        matches_unc, u_track_unc, u_detection_unc = self._handle_unconfirmed_tracks(u_detection_first, detections, activated_stracks, removed_stracks, unconfirmed)

        # Initialize new tracks 初始化新的追踪
        self._initialize_new_tracks(u_detection_unc, activated_stracks, [detections[i] for i in u_detection_first])

        # Update lost and removed tracks
        self._update_track_states(lost_stracks, removed_stracks)

        # Merge and prepare output
        return self._prepare_output(activated_stracks, refind_stracks, lost_stracks, removed_stracks)

    def _split_detections(self, dets, embs):
        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        confs = dets[:, 4]
        second_mask = np.logical_and(confs > self.track_low_thresh, confs < self.track_high_thresh)
        dets_second = dets[second_mask]
        first_mask = confs > self.track_high_thresh
        dets_first = dets[first_mask]
        embs_first = embs[first_mask] if embs is not None else None
        return dets, dets_first, embs_first, dets_second

    def _create_detections(self, dets_first, features_high):
        if len(dets_first) > 0:
            if self.with_reid:
                detections = [STrack(det, f, max_obs=self.max_obs) for (det, f) in zip(dets_first, features_high)]  #
            else:
                detections = [STrack(det, max_obs=self.max_obs) for det in dets_first]
        else:
            detections = []
        return detections

    def _separate_tracks(self):
        unconfirmed, active_tracks = [], []
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                active_tracks.append(track)
        return unconfirmed, active_tracks

    def _first_association(self, dets, dets_first, active_tracks, unconfirmed, img, detections, activated_stracks, refind_stracks, strack_pool):
        # 第一次关联
        STrack.multi_predict(strack_pool)  # 批量处理

        # Fix camera motion
        warp = self.cmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Associate with high confidence detection boxes
        ious_dists = iou_distance(strack_pool, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh
        if self.fuse_first_associate:
            ious_dists = fuse_score(ious_dists, detections)

        if self.with_reid:
            emb_dists = embedding_distance(strack_pool, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)
                
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_count)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)
                
        return matches, u_track, u_detection

    def _second_association(self, dets_second, activated_stracks, lost_stracks, refind_stracks, u_track_first, strack_pool):
        if len(dets_second) > 0:
            detections_second = [STrack(det, max_obs=self.max_obs) for det in dets_second]
        else:
            detections_second = []

        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track_first
            if strack_pool[i].state == TrackState.Tracked
        ]

        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection = linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
                
        return matches, u_track, u_detection


    def _handle_unconfirmed_tracks(self, u_detection, detections, activated_stracks, removed_stracks, unconfirmed):
        """
        Handle unconfirmed tracks (tracks with only one detection frame).

        Args:
            u_detection: Unconfirmed detection indices.
            detections: Current list of detections.
            activated_stracks: List of newly activated tracks.
            removed_stracks: List of tracks to remove.
        """
        # Only use detections that are unconfirmed (filtered by u_detection)
        detections = [detections[i] for i in u_detection]
        
        # Calculate IoU distance between unconfirmed tracks and detections
        ious_dists = iou_distance(unconfirmed, detections)
        
        # Apply IoU mask to filter out distances that exceed proximity threshold
        ious_dists_mask = ious_dists > self.proximity_thresh
        ious_dists = fuse_score(ious_dists, detections)
        
        # Fuse scores for IoU-based and embedding-based matching (if applicable)
        if self.with_reid:
            emb_dists = embedding_distance(unconfirmed, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0  # Apply the IoU mask to embedding distances
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        # Perform data association using linear assignment on the combined distances
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        
        # Update matched unconfirmed tracks
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_count)
            activated_stracks.append(unconfirmed[itracked])

        # Mark unmatched unconfirmed tracks as removed
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
            
        return matches, u_unconfirmed, u_detection

    def _initialize_new_tracks(self, u_detections, activated_stracks, detections):
        for inew in u_detections:
            track = detections[inew]
            if track.conf < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_count)
            activated_stracks.append(track)

    def _update_tracks(self, matches, strack_pool, detections, activated_stracks, refind_stracks, mark_removed=False):
        # Update or reactivate matched tracks
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)
        
        # Mark only unmatched tracks as removed, if mark_removed flag is True
        if mark_removed:
            unmatched_tracks = [strack_pool[i] for i in range(len(strack_pool)) if i not in [m[0] for m in matches]]
            for track in unmatched_tracks:
                track.mark_removed()

    def _update_track_states(self, lost_stracks, removed_stracks):
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

    def _prepare_output(self, activated_stracks, refind_stracks, lost_stracks, removed_stracks):
        self.active_tracks = [
            t for t in self.active_tracks if t.state == TrackState.Tracked
        ]
        self.active_tracks = joint_stracks(self.active_tracks, activated_stracks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(
            self.active_tracks, self.lost_stracks
        )

        outputs = [
            [*t.xyxy, t.id, t.conf, t.cls, t.det_ind]
            for t in self.active_tracks if t.is_activated
        ]

        return np.asarray(outputs)
