"""
作者：黄瀚扬
时间：2024/11/26

"""
import torch
import os
import streamlit as st
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from boxmot import BotSort, StrongSort
import json


def get_result_video_path(original_video_name, video_folder):
    """根据原始视频文件名生成结果视频的完整路径"""
    # 假设原视频名字为: 20241129-093743_rack-1_left_layer-1_RGB.mp4
    parts = original_video_name.split('_')  # 按下划线分隔
    date_info = parts[0]  # 提取日期部分: 20241129
    time_info = parts[1]  # 提取时间部分: 093743
    location_info = "_".join(parts[2:])  # 提取剩余部分作为位置描述: rack-1_left_layer-1_RGB
    # 查找当前文件夹中包含日期和时间部分的文件，作为前缀
    matching_files = [f for f in os.listdir(video_folder) if f"{date_info}_{time_info}" in f]
    # 如果找到匹配的文件，取第一个文件名作为前缀
    prefix = matching_files[0] if matching_files else "unknown_prefix"
    # 构建结果视频文件名
    result_video_name = f"{prefix}_{date_info}_{time_info}_{location_info.replace('RGB', '')}.mp4"
    # 结果视频的完整路径（假设保存在一个结果文件夹中）
    result_folder = os.path.join(video_folder, "results")  # 假设结果保存在该文件夹
    result_video_path = os.path.join(result_folder, result_video_name)

    return result_video_path


def preview_video(video_path):
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)  # 使用st.video函数播放视频


# 计算总计数
def calculate_total_count(class_counts):
    return sum(class_counts.values())


def load_json_info(video_path):
    """
    加载与视频同路径的 JSON 文件。
    """
    json_path = Path(video_path).with_suffix('.json')
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# 添加显示统计结果的表格
def display_statistics_as_table(class_counts):
    # 将类别和数量转换为 DataFrame
    df = pd.DataFrame(list(class_counts.items()), columns=["类别", "数量"])
    # 显示表格
    st.table(df)


# 加载 YOLO 模型和追踪器
# @st.cache_resource
def load_model(tracker_type, show_details=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model = YOLO('tracking/weights/yolov8l_bestmodel_dataset3131_cls7_416_416_renamecls.pt')
    if show_details:
        st.write("det模型加载完毕")
    yolo_model.to(device)
    if tracker_type == "BotSort":  # 选择追踪器 BotSort
        tracker = BotSort(
            reid_weights=Path('tracking/weights/resnet50_berry_add_6.pt'),
            device=0,
            half=False,
            track_high_thresh=0.6
        )
        print("BotSort Initialization completed")
        if show_details:
            st.write("BotSort Initialization completed")
    elif tracker_type == "StrongSort":  # 选择追踪器 StrongSort
        tracker = StrongSort(
            reid_weights=Path('tracking/weights/resnet50_berry_add_6.pt'),  # Path to ReID model
            device=0,  # Use CPU for inference
            half=False,
            max_cos_dist=0.4
        )
        print("StrongSort Initialization completed")
        if show_details:
            st.write("StrongSort Initialization completed")
    else:  # 其他追踪器
        print("no tracker")
        if show_details:
            st.write("no tracker")
        tracker = []
    return yolo_model, tracker


def save_tracking_results(save_dir, file_name, tracking_results):
    """
    保存追踪结果到指定路径和文件。
    """
    save_path = Path(save_dir) / file_name
    save_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    with open(save_path, "w") as f:
        for line in tracking_results:
            f.write(line + "\n")
    return save_path


# 定义函数：保存统计结果
def save_statistics(save_dir, file_name, total_count, class_counts):
    """
    保存统计结果到指定路径和文件。
    """
    save_path = Path(save_dir) / file_name
    save_path.parent.mkdir(parents=True, exist_ok=True)  # 创建文件夹
    with open(save_path, "w") as f:
        f.write(f"总果实数量: {total_count}\n")
        for class_name, count in class_counts.items():
            f.write(f"{class_name}: {count}\n")
    return save_path

