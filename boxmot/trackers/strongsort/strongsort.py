# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
from torch import device
from pathlib import Path

from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.motion.cmc import get_cmc_method
from boxmot.trackers.strongsort.sort.detection import Detection
from boxmot.trackers.strongsort.sort.tracker import Tracker
from boxmot.utils.matching import NearestNeighborDistanceMetric
from boxmot.utils.ops import xyxy2tlwh
from boxmot.utils import logger
from boxmot.trackers.basetracker import BaseTracker


class StrongSort(object):
    """
    StrongSORT Tracker: A tracking algorithm that utilizes a combination of appearance and motion-based tracking.

    Args:
        model_weights (str): Path to the model weights for ReID (Re-Identification).
        device (str): Device on which to run the model (e.g., 'cpu' or 'cuda').
        fp16 (bool): Whether to use half-precision (fp16) for faster inference on compatible devices.
        per_class (bool, optional): Whether to perform per-class tracking. If True, tracks are maintained separately for each object class.
        max_dist (float, optional): Maximum cosine distance for ReID feature matching in Nearest Neighbor Distance Metric.
        max_iou_dist (float, optional): Maximum Intersection over Union (IoU) distance for data association. Controls the maximum allowed distance between tracklets and detections for a match.
        max_age (int, optional): Maximum number of frames to keep a track alive without any detections.
        n_init (int, optional): Number of consecutive frames required to confirm a track.
        nn_budget (int, optional): Maximum size of the feature library for Nearest Neighbor Distance Metric. If the library size exceeds this value, the oldest features are removed.
        mc_lambda (float, optional): Weight for motion consistency in the track state estimation. Higher values give more weight to motion information.
        ema_alpha (float, optional): Alpha value for exponential moving average (EMA) update of appearance features. Controls the contribution of new and old embeddings in the ReID model.
    """
    def __init__(
        self,
        reid_weights: Path,
        device: device,
        half: bool,
        per_class: bool = False,
        max_cos_dist=0.2,  #
        max_iou_dist=0.7,  #
        max_age=30,  #
        n_init=3,  #
        nn_budget=100,  #
        mc_lambda=0.98,  #
        ema_alpha=0.9,  #
    ):

        self.per_class = per_class
        self.model = ReidAutoBackend(
            weights=reid_weights, device=device, half=half
        ).model

        self.tracker = Tracker(
            metric=NearestNeighborDistanceMetric("cosine", max_cos_dist, nn_budget),
            max_iou_dist=max_iou_dist,
            max_age=max_age,
            n_init=n_init,
            mc_lambda=mc_lambda,
            ema_alpha=ema_alpha,
        )
        self.cmc = get_cmc_method('ecc')()

    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:  # 更新
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        logger.debug(f"det_count: {dets.shape[0]}")  # log输出每帧结果
        # print("det_count:", dets.shape[0])
        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])  # 堆叠 多加了一列 序列
        xyxy = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]
        det_ind = dets[:, 6]

        if len(self.tracker.tracks) >= 1:  #
            warp_matrix = self.cmc.apply(img, xyxy)  # 计算图像运动
            for track in self.tracker.tracks:
                track.camera_update(warp_matrix)  # 在每个轨迹上加上相机运动

        # extract appearance information for each detection
        if embs is not None:
            features = embs
        else:
            features = self.model.get_features(xyxy, img)  # 计算reid相似度

        tlwh = xyxy2tlwh(xyxy)  # 转化为top left weight height
        detections = [
            Detection(box, conf, cls, det_ind, feat) for
            box, conf, cls, det_ind, feat in
            zip(tlwh, confs, clss, det_ind, features)
        ]  # 打包合并 存入det

        # update tracker
        self.tracker.predict()  # 更新轨迹状态
        self.tracker.update(detections)  # 将det和轨迹进行匹配

        # output bbox identities box id
        outputs = []
        for track in self.tracker.tracks:  # 遍历每个轨迹 将匹配上的输出
            if not track.is_confirmed() or track.time_since_update >= 1:  # 不是确认的轨迹 或者是 丢失的轨迹
                continue

            x1, y1, x2, y2 = track.to_tlbr()  #

            id = track.id
            conf = track.conf
            cls = track.cls
            det_ind = track.det_ind

            outputs.append(
                np.concatenate(([x1, y1, x2, y2], [id], [conf], [cls], [det_ind])).reshape(1, -1)
            )
        if len(outputs) > 0:
            return np.concatenate(outputs)  # 返回
        return np.array([])
