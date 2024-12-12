import streamlit as st
import platform
import cv2
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from streamlit_utils.utils import *


# 全局变量定义
classes = ['Ripe', 'Ripe7', 'Ripe4', 'Ripe2', 'Unripe', 'Flower', 'Disease']
class_counts = {cls: 0 for cls in classes}
total_count = 0

# 根据操作系统选择路径列表
is_windows = platform.system() == "Windows"


# 定义函数：处理检测和追踪
def process_frame(frame, yolo_model, tracker, conf_thresh, iou_thresh, use_mask_opt, track_id_set):
    """
    处理单帧图像，包括检测和追踪。
    """
    global total_count, class_counts

    # YOLOv8 检测
    results = yolo_model(frame, conf=conf_thresh, iou=iou_thresh, agnostic_nms=True, imgsz=640,
                         classes=[[0, 1, 2, 3, 4, 6]])
    dets, reid_masks = [], None

    if results is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()
            if conf >= 0.1:
                dets.append([x1, y1, x2, y2, conf, cls])
        dets = np.array(dets)
        if results[0].masks is not None:
            reid_masks = np.array([mask for mask in results[0].masks.data.cpu().numpy()])  # mask(0, 1)

    # 更新追踪器
    tracked_objects = tracker.update(dets, frame, reid_masks) if use_mask_opt else tracker.update(dets, frame)

    # 统计追踪结果
    for obj in tracked_objects:
        bbox, cls, track_id = obj[0:4], int(obj[6]), int(obj[4])
        class_name = classes[cls]
        if track_id and track_id not in track_id_set:
            track_id_set.add(track_id)
            total_count += 1
            class_counts[class_name] += 1
    return tracked_objects


def process_video(video_path, yolo_model, tracker, conf_thresh, iou_thresh, use_mask_opt, processed_video_placeholder):
    """
    处理视频并统计追踪结果。
    """
    global total_count, class_counts
    vid = cv2.VideoCapture(video_path)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    output_path = 'processed_output.mp4'
    out_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'H264'), fps, (frame_width, frame_height))
    track_id_set = set()
    # 保存追踪结果相关变量
    texts = []  # 用于存储追踪结果的每一行

    stframe, stframetxt, progress_bar = st.empty(), st.empty(), st.empty()
    frame_id = 0
    while True:
        ret, frame = vid.read()
        if not ret:
            st.write("处理完成")
            break
        progress_bar.progress((frame_id + 1) / total_frames, '进度')  # 添加进度条
        # st.spinner('加载中...')
        stframetxt.text(f"总帧数: {total_frames}   正在处理帧 {frame_id}")
        # 处理每一帧
        tracked_objects = process_frame(frame, yolo_model, tracker, conf_thresh, iou_thresh, use_mask_opt, track_id_set)

        # 绘制追踪结果
        if tracker.__class__.__name__ == "BotSort":
            tracker.plot_results(frame, show_trajectories=True)
        out_video.write(frame)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # 保存追踪结果到 texts 列表
        for re in tracked_objects:
            bbox = re[:4]
            track_id = int(re[4])
            cls = int(re[6])
            class_name = classes[cls]
            # 如果是新 track_id，更新计数
            if track_id not in track_id_set:
                track_id_set.add(track_id)
                total_count += 1
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            # 构造保存行
            line = (frame_id, class_name + '_', track_id, int(bbox[0]), int(bbox[1]),
                    int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]), -1, -1, -1, 0)
            texts.append("%g,%s,%g,%g,%g,%g,%g,%g,%g,%g,%g" % line)
        frame_id += 1
    vid.release()
    out_video.release()
    # 提供视频下载链接
    with open(output_path, "rb") as f:
        st.download_button("下载处理后的视频", f, file_name="result.mp4")
    processed_video_placeholder.video(output_path)


def get_video_metadata(video_paths):
    """从视频文件名中提取场景和日期信息"""
    metadata = []
    for path in video_paths:
        filename = Path(path).stem  # 获取文件名不带扩展名
        parts = filename.split('_')
        if len(parts) >= 3 and parts[0] == 'aiwei':
            scene = "艾维"
            date = parts[1]
            metadata.append((scene, date, path))
    return metadata


def main():
    st.title("巡检统计分析结果")
    st.sidebar.title("数据选择")
    # 假设所有视频在一个文件夹下
    video_dir = r'D:\华毅\目标追踪数据集\test' if platform.system() == "Windows" else r'/home/xplv/huanghanyang/Track_Datasets/test'
    video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
    # 获取视频元数据
    metadata = get_video_metadata(video_paths)
    # 提取唯一的场景和日期
    scenes = list(set([m[0] for m in metadata]))
    dates = list(set([m[1] for m in metadata]))
    # 场景和日期选择
    scene_option = st.sidebar.selectbox("选择场景", scenes)
    # date_option = st.sidebar.selectbox("选择日期", sorted(dates, reverse=True))
    # 日期选择模块
    selected_date = st.sidebar.date_input("选择日期", value=datetime(2024, 12, 5))
    selected_date_str = selected_date.strftime('%Y%m%d')  # 转换为字符串格式
    # 根据场景和日期筛选视频路径，并准备只显示文件名
    video_paths_filtered = [m[2] for m in metadata if m[0] == scene_option and m[1] == selected_date_str]
    video_names_filtered = [Path(path).name for path in video_paths_filtered]  # 只显示文件名
    # 视频选择
    selected_video_name = st.sidebar.selectbox("选择种植架", video_names_filtered)
    video_path = video_paths_filtered[video_names_filtered.index(selected_video_name)] if video_names_filtered else None
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    with col1:
        st.subheader("原始视频")
        original_video_placeholder = st.empty()
        if video_path:
            original_video_placeholder.video(open(video_path, 'rb').read())
    with col2:
        st.subheader("结果视频")
        processed_video_placeholder = st.empty()

    with col3:
        st.subheader("视频的基本信息")
        json_placeholder = st.empty()
        # 加载 JSON 文件
        if video_path:
            video_info = load_json_info(video_path)
            json_placeholder.json(video_info)  # 显示 JSON 信息
        else:
            json_placeholder.text("未找到基本信息文件。")

    with col4:
        st.subheader("视频追踪结果")
        tracking_results_placeholder = st.empty()

        # 初始化第一个表格：类别计数
        df_classes = pd.DataFrame(list(class_counts.items()), columns=["类别", "数量"])
        # 添加总计到第一个表格
        total_count = calculate_total_count(class_counts)
        df_classes.loc[len(df_classes)] = ["总计", total_count]

        # 显示第一个表格
        st.subheader("类别统计表")
        st.table(df_classes)

        # 第二个表格示例：其他统计（例如帧数或其他追踪结果）
        additional_data = {
            "指标": ["总帧数", "检测数量", "追踪数量"],
            "数值": [1, 1, 1]
        }
        df_additional = pd.DataFrame(additional_data)

        # 显示第二个表格
        st.subheader("其他统计表")
        st.table(df_additional)

        # 如果有更多表格，可以按照类似方式添加
        st.subheader("其他信息")
        st.text("在此处添加其他统计或分析结果。")

    if video_path:
        st.write(f"已选择数据: {selected_video_name}")
        start_button = st.button("开始处理")
        if start_button:
            yolo_model, tracker = load_model("BotSort")
            process_video(
                video_path=video_path,
                yolo_model=yolo_model,
                tracker=tracker,
                conf_thresh=0.1,
                iou_thresh=0.7,
                use_mask_opt=False,
                processed_video_placeholder=processed_video_placeholder
            )

            # 更新表格数据和总计
            df = pd.DataFrame(list(class_counts.items()), columns=["类别", "数量"])
            total_count = calculate_total_count(class_counts)
            df.loc[len(df)] = ["总计", total_count]
            tracking_results_placeholder.table(df)


def preview_video(video_path):
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)  # 使用st.video函数播放视频


# 程序入口
if __name__ == "__main__":
    main()
