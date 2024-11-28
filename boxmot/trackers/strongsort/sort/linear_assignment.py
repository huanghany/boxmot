# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from __future__ import absolute_import

import numpy as np
from scipy.optimize import linear_sum_assignment

from boxmot.utils.matching import chi2inv95

INFTY_COST = 1e5


def min_cost_matching(
    distance_metric,
    max_distance,  # 0.4
    tracks,
    detections,
    track_indices=None,
    detection_indices=None,
):
    """Solve linear assignment problem. 线性分配
    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:  # 轨迹或det为0
        return [], track_indices, detection_indices  # Nothing to match. 都没匹配上

    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)  # 使用iou进行计算
    print("iou dist:")
    print(cost_matrix)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5  # 大于阈值(0.7)的赋很大值
    row_indices, col_indices = linear_sum_assignment(cost_matrix)  # 获取匹配上的行列 row行 col列

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:  # 列没在序列中
            unmatched_detections.append(detection_idx)  # 没匹配上的det
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:  # 行没在
            unmatched_tracks.append(track_idx)  # 没匹配上的轨迹
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:  # 大于阈值0.7的不匹配
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))  # 匹配上
    return matches, unmatched_tracks, unmatched_detections  #


def matching_cascade(  # 级联匹配
    distance_metric,
    max_distance,
    cascade_depth,
    tracks,  # 已有轨迹
    detections,  # 输入det
    track_indices=None,  # 输入det
    detection_indices=None,  # 输入轨迹
):
    """Run matching cascade.
    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.
    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))  #

    unmatched_detections = detection_indices  # 待匹配det
    matches = []  # 初始化matches
    track_indices_l = [k for k in track_indices]  # 待匹配轨迹
    matches_l, _, unmatched_detections = min_cost_matching(  # 调用相似度计算 轨迹和det匹配
        distance_metric,  # 距离度量函数
        max_distance,  # 最大距离阈值
        tracks,
        detections,
        track_indices_l,  # 待匹配轨迹
        unmatched_detections,  # 待匹配det
    )
    matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))  # 没有匹配上的轨迹
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(  # 门控过滤 过滤匹配矩阵中的无效匹配
    cost_matrix,  # 匹配矩阵
    tracks,
    detections,
    track_indices,  # 轨迹
    detection_indices,  # det
    mc_lambda,  # 0.995
    gated_cost=INFTY_COST,  # 100000 用于标记不可行的关联
    only_position=False,  # 门控信息是否仅基于位置信息(x,y) 如果为True，则忽略尺度和宽高比
):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.
    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.
    Returns
    -------
    ndarray
        Returns the modified cost matrix.
    """
    
    gating_threshold = chi2inv95[4]  # 卡方分布下95%水平的阈值 自由度为4  9.4877 因为是四维数组
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])  # 转换格式(x y ratio h)
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = track.kf.gating_distance(   # 计算kl门控距离
            track.mean,
            track.covariance,
            measurements,
            only_position
        )
        print("gating_distance门控距离: ")
        print(gating_distance)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost   # 大于与阈值9.4877的设为100000(无效)
        print("原代价矩阵：")
        print(cost_matrix)
        cost_matrix[row] = (
            mc_lambda * cost_matrix[row] + (1 - mc_lambda) * gating_distance  # x原代价++（1-x)门控距离 0.98x原+0.02x门
        )  # 混合调整初始代价和门控距离
        print("新代价矩阵：")
        print(cost_matrix)
    return cost_matrix
