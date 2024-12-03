"""
作者：黄瀚扬
时间：2024/11/26

"""
import torch
import streamlit as st
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from boxmot import BotSort, StrongSort


# 添加显示统计结果的表格
def display_statistics_as_table(class_counts):
    # 将类别和数量转换为 DataFrame
    df = pd.DataFrame(list(class_counts.items()), columns=["类别", "数量"])

    # 显示表格
    st.table(df)


# 加载 YOLO 模型和追踪器
# @st.cache_resource
def load_model(tracker_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model = YOLO('tracking/weights/yolov8l_bestmodel_dataset3131_cls7_416_416_renamecls.pt')
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
        st.write("BotSort Initialization completed")
    elif tracker_type == "StrongSort":  # 选择追踪器 StrongSort
        tracker = StrongSort(
            reid_weights=Path('tracking/weights/resnet50_berry_add_6.pt'),  # Path to ReID model
            device=0,  # Use CPU for inference
            half=False,
            max_cos_dist=0.4
        )
        print("StrongSort Initialization completed")
        st.write("StrongSort Initialization completed")
    else:  # 其他追踪器
        print("no tracker")
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

