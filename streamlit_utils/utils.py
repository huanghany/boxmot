"""
作者：黄瀚扬
时间：2024/11/26

"""
import torch
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from boxmot import BotSort
from boxmot.utils import TRACKER_CONFIGS
import tempfile
import platform


# 添加显示统计结果的表格
def display_statistics_as_table(class_counts):
    # 将类别和数量转换为 DataFrame
    df = pd.DataFrame(list(class_counts.items()), columns=["类别", "数量"])

    # 显示表格
    st.table(df)


# 加载 YOLO 模型和追踪器
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model = YOLO('tracking/weights/yolov8l_bestmodel_dataset3131_cls7_416_416_renamecls.pt')
    yolo_model.to(device)
    tracker = BotSort(
        reid_weights=Path('tracking/weights/resnet50_berry_add_6.pt'),
        device=0,
        half=False,
        track_high_thresh=0.6
    )
    return yolo_model, tracker

