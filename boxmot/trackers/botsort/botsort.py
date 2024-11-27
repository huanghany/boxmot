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
            track_high_thresh: float = 0.5,  # default=0.6
            track_low_thresh: float = 0.1,  #
            new_track_thresh: float = 0.6,  #
            track_buffer: int = 30,
            match_thresh: float = 0.8,
            proximity_thresh: float = 0.5,
            appearance_thresh: float = 0.25,
            cmc_method: str = "ecc",
            frame_rate=30,
            fuse_first_associate: bool = False,  # 第一次关联时是否结合运动和外观
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
        self.kalman_filter = KalmanFilterXYWH()  # 卡尔曼

        # ReID module
        self.proximity_thresh = proximity_thresh  # 第一次关联的iou阈值
        self.appearance_thresh = appearance_thresh  # 相似度关联阈值
        self.with_reid = with_reid  # 是否使用reid模型
        # self.with_reid = False
        if self.with_reid:  # 使用reid
            self.model = ReidAutoBackend(
                weights=reid_weights, device=device, half=half
            ).model

        self.cmc = get_cmc_method(cmc_method)()
        self.fuse_first_associate = fuse_first_associate

    @BaseTracker.on_first_frame_setup
    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None, masks: np.ndarray = None) -> np.ndarray:  # 更新追踪器
        # emb 为特征
        masks = embs.copy() if embs is not None else None
        embs = None
        print("det_count:", dets.shape[0])
        self.check_inputs(dets, img)  #
        self.frame_count += 1
        activated_stracks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        # Preprocess detections det预处理
        dets, dets_first, embs_first, dets_second = self._split_detections(dets, embs)  #

        # Extract appearance features 提取外观(reid)特征
        if self.with_reid and masks is not None:
            features_high = self.model.get_features_mask(dets_first[:, 0:4], img, masks)  # 获取相似度(mask)
        elif self.with_reid and embs is None:
            features_high = self.model.get_features(dets_first[:, 0:4], img)  # 获取相似度(bbox)
        else:
            features_high = embs_first if embs_first is not None else []

        # Create detections 创建追踪
        detections = self._create_detections(dets_first, features_high)
        # 分数高的追踪
        # Separate unconfirmed and active tracks 将已经追踪上的和没追踪上的分开
        unconfirmed, active_tracks = self._separate_tracks()  # 刚刚创建的轨迹 被激活过的轨迹
        # 刚刚创建的轨迹(除了第一帧)
        strack_pool = joint_stracks(active_tracks, self.lost_stracks)  # 被追踪
        # 被追踪过的轨迹
        # First association 第一次关联 (conf>track_high_thresh的det和被激活过的轨迹匹配)
        matches_first, u_track_first, u_detection_first = self._first_association(dets, dets_first, active_tracks,
                                                                                  unconfirmed, img, detections,
                                                                                  activated_stracks, refind_stracks,
                                                                                  strack_pool)

        # Second association 第二次关联 (conf<track_high_thresh的det和被激活过的而且第一次关联时没有匹配上的轨迹匹配)
        matches_second, u_track_second, u_detection_second = self._second_association(dets_second, activated_stracks,
                                                                                      lost_stracks, refind_stracks,
                                                                                      u_track_first, strack_pool)

        # Handle unconfirmed tracks (将第一次关联没有匹配上的det和上一帧刚刚创建的轨迹匹配,没匹配上就创建新轨迹)
        matches_unc, u_track_unc, u_detection_unc = self._handle_unconfirmed_tracks(u_detection_first, detections,
                                                                                    activated_stracks, removed_stracks,
                                                                                    unconfirmed)

        # Initialize new tracks 初始化新的追踪
        self._initialize_new_tracks(u_detection_unc, activated_stracks, [detections[i] for i in u_detection_first])
        # 为第一次和handle没有关联上的创建新轨迹 conf要>new_track_thresh(0.6)
        # Update lost and removed tracks 处理超时的追踪
        self._update_track_states(lost_stracks, removed_stracks)

        # Merge and prepare output
        return self._prepare_output(activated_stracks, refind_stracks, lost_stracks, removed_stracks)

    def _split_detections(self, dets, embs):
        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])  # 新增编号 为每个框添加ID
        confs = dets[:, 4]  # 添加置信度
        second_mask = np.logical_and(confs > self.track_low_thresh,
                                     confs < self.track_high_thresh)  # 标记出置信度在track_low_thresh之间的(0.1-0.6)
        dets_second = dets[second_mask]  # 中间置信度的放入第二次det
        first_mask = confs > self.track_high_thresh
        dets_first = dets[first_mask]  # 置信度高于track_high_thresh(0.6)的放入第一次det
        embs_first = embs[first_mask] if embs is not None else None  # embs不空的话将置信度高的那部分放入
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
                unconfirmed.append(track)  # 前一帧没被追踪
            else:
                active_tracks.append(track)  # 前一帧追踪
        return unconfirmed, active_tracks

    def _first_association(self, dets, dets_first, active_tracks, unconfirmed, img, detections, activated_stracks,
                           refind_stracks, strack_pool):
        # 第一次关联
        STrack.multi_predict(strack_pool)  # 批量计算卡尔曼滤波

        # Fix camera motion  相机运动修正 cmc
        warp = self.cmc.apply(img, dets)  # 减少相机运动带来误差
        STrack.multi_gmc(strack_pool, warp)  # 计算轨迹的相机移动偏差
        STrack.multi_gmc(unconfirmed, warp)  # 计算轨迹的相机移动偏差

        # Associate with high confidence detection boxes 关联高置信度检测框
        ious_dists = iou_distance(strack_pool, detections)  # 计算iou距离
        if strack_pool and detections:
            for st in strack_pool:
                # print(st.mean)
                predict = st.xyxy.copy()
                kl = [predict[0], predict[1], predict[2] - predict[0], predict[3] - predict[1]]
                kl = [round(x, 2) for x in kl]
                # print("kl:", kl)
            for st in detections:
                predict = st.xyxy.copy()
                det_1 = [predict[0], predict[1], predict[2] - predict[0], predict[3] - predict[1]]
                det_1 = [round(x, 2) for x in det_1]
                # print("det:", det_1)
        # values_less_than_one = ious_dists[ious_dists < 1]
        # print("iou_dists 中小于 1 的值有：")
        # print(values_less_than_one)
        ious_dists_mask = ious_dists > self.proximity_thresh  # 大于iou阈值的不采用
        if self.fuse_first_associate:  # 是否融合置信度 false
            ious_dists = fuse_score(ious_dists, detections)

        if self.with_reid:  # 基于reid计算
            emb_dists = embedding_distance(strack_pool, detections) / 2.0  #
            # print(emb_dists)
            emb_dists_1 = emb_dists.copy()
            values_less_than_one = emb_dists_1[emb_dists_1 < 1]
            print("emb_dists 中小于 1 的值有：")
            print(values_less_than_one)
            emb_dists[emb_dists > self.appearance_thresh] = 1.0  # 相似度高于阈值的不采用

            change_1 = True
            if change_1:
                vaild_mask = emb_dists <= 0.2  # reid距离小于0.2时
                state_mask = np.array([track.state == TrackState.Lost for track in strack_pool], dtype=bool)  # 是丢失的轨迹
                state_mask = state_mask[:, np.newaxis]  # 拓展维度
                ious_dists_mask_2 = ious_dists < 1  # iou距离小于1
                combied_mask = vaild_mask & state_mask & ious_dists_mask_2  # 合并
                ious_dists_mask[combied_mask] = False  # 添加掩码
            emb_dists[ious_dists_mask] = 1.0  # iou满足的时候emb_dist也为1
            dists = np.minimum(ious_dists, emb_dists)  # 取iou和相似度的最小作为dist
        else:
            dists = ious_dists

        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)  # 进行匹配 0.8
        if len(matches) > 0:
            print(f"first_match: {len(matches)}")
            # print("first_iou_dist:")
            # print(ious_dists)
            # print("first_emb_dist: ")
            # print(emb_dists_1)
        if len(u_detection) > 0:
            print(f"first_not_match: {len(u_detection)}")
            if len(u_track) == 0:
                print("no track")
            else:
                print("first_iou_dist:")
                # print(ious_dists)
                print("first_emb_dist: ")
                # print(emb_dists_1)
        if len(u_track) > 0:
            print(f"first_track_not_match: {len(u_track)}")
        # dists为不同轨迹和检测目标之间的距离 距离小于0.8的进行匹配
        for itracked, idet in matches:
            track = strack_pool[itracked]  # 已追踪的对象
            det = detections[idet]
            if track.state == TrackState.Tracked:  # 状态为已追踪 更新状态
                track.update(detections[idet], self.frame_count)  #
                activated_stracks.append(track)  #
            else:  # 重新追踪
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        return matches, u_track, u_detection  #

    def _second_association(self, dets_second, activated_stracks, lost_stracks, refind_stracks, u_track_first,
                            strack_pool):
        if len(dets_second) > 0:
            detections_second = [STrack(det, max_obs=self.max_obs) for det in dets_second]
        else:
            detections_second = []

        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track_first  # 用第一次没追踪上的轨迹
            if strack_pool[i].state == TrackState.Tracked  # 是被追踪的状态
        ]

        dists = iou_distance(r_tracked_stracks, detections_second)  # 直接用iou阈值
        # values_less_than_one = dists[dists < 1]
        # print("iou_dists 中小于 1 的值有：")
        # print(values_less_than_one)
        matches, u_track, u_detection = linear_assignment(dists, thresh=0.5)  # 0.5 ?
        if len(matches) > 0:
            print(f"second_match: {len(matches)}")
            # print(dists)
        if len(u_detection) > 0:
            print(f"second_not_match: {len(u_detection)}")
            if len(u_track) == 0:
                print("no track")
            else:
                print("dists:")
                # print(dists)
        for itracked, idet in matches:  # 匹配上的
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        for it in u_track:  # 未被匹配上的轨迹
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:  # 如果状态不是丢失设为丢失
                track.mark_lost()
                lost_stracks.append(track)

        return matches, u_track, u_detection

    def _handle_unconfirmed_tracks(self, u_detection, detections, activated_stracks, removed_stracks, unconfirmed):
        """
        Handle unconfirmed tracks (tracks with only one detection frame).

        Args:
            u_detection: Unconfirmed detection indices. 没被追踪的det
            detections: Current list of detections.
            activated_stracks: List of newly activated tracks.
            removed_stracks: List of tracks to remove.
        """
        # Only use detections that are unconfirmed (filtered by u_detection)
        detections = [detections[i] for i in u_detection]

        # Calculate IoU distance between unconfirmed tracks and detections
        ious_dists = iou_distance(unconfirmed, detections)  # 计算iou
        # values_less_than_one = ious_dists[ious_dists < 1]
        # print("iou_dists 中小于 1 的值有：")
        # print(values_less_than_one)
        # Apply IoU mask to filter out distances that exceed proximity threshold
        ious_dists_mask = ious_dists > self.proximity_thresh  #
        ious_dists = fuse_score(ious_dists, detections)  # 融合置信度分数(不应该添加?)(置信度低时距离更大)

        # Fuse scores for IoU-based and embedding-based matching (if applicable)
        if self.with_reid:
            emb_dists = embedding_distance(unconfirmed, detections) / 2.0
            emb_dists_1 = emb_dists.copy()
            values_less_than_one = emb_dists_1[emb_dists_1 < 1]
            print("emb_dists 中小于 1 的值有：")
            print(values_less_than_one)
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0  # Apply the IoU mask to embedding distances
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        # Perform data association using linear assignment on the combined distances
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)  #
        if len(matches) > 0:
            print(f"handle_match, new tracker: {len(matches)}")
        if len(u_detection) > 0:
            print(f"handle_not_match, new tracker: {len(u_detection)}")
            if len(u_unconfirmed) == 0:
                print('create new track')
            else:
                print("handle_iou_dist:")
                # print(ious_dists)
                print("handle_emb_dist: ")
                # print(emb_dists_1)
        # Update matched unconfirmed tracks
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_count)
            activated_stracks.append(unconfirmed[itracked])

        # Mark unmatched unconfirmed tracks as removed
        for it in u_unconfirmed:  # 没匹配上的轨迹就remove
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        return matches, u_unconfirmed, u_detection

    def _initialize_new_tracks(self, u_detections, activated_stracks, detections):  # u为未与现有轨迹匹配的检测目标
        for inew in u_detections:  # handle没有关联上的
            track = detections[inew]
            if track.conf < self.new_track_thresh:  # 置信度低于阈值
                # print(f"conf= {track.conf} no new tracker")
                continue  # 过滤低置信度

            track.activate(self.kalman_filter, self.frame_count)  # 激活轨迹
            activated_stracks.append(track)  # 添加

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
        )  #

        outputs = [
            [*t.xyxy, t.id, t.conf, t.cls, t.det_ind]
            for t in self.active_tracks if t.is_activated
        ]

        return np.asarray(outputs)
