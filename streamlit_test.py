import torch
import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import BotSort
from boxmot.utils import TRACKER_CONFIGS
import tempfile

# 定义全局统计变量
total_count = 0
class_counts = {
    "Unripe": 0,
    "Ripe": 0,
    "Ripe7": 0,
    "Ripe4": 0,
    "Ripe2": 0,
    "Flower": 0,
    "Disease": 0
}
classes = ['Ripe', 'Ripe7', 'Ripe4', 'Ripe2', 'Unripe', 'Flower', 'Disease']

# Streamlit 界面
st.title("YOLOv8 和 BoT-SORT 视频追踪")
st.sidebar.title("参数设置")
uploaded_video = st.sidebar.file_uploader("上传视频文件", type=["mp4", "avi", "mov"])
conf_thresh = st.sidebar.slider("置信度阈值", 0.0, 1.0, 0.1, step=0.01)
iou_thresh = st.sidebar.slider("IOU 阈值", 0.0, 1.0, 0.7, step=0.01)
save_txt_opt = st.sidebar.checkbox("保存结果到文本文件", value=False)

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

yolo_model, tracker = load_model()

# 视频处理
if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name

    vid = cv2.VideoCapture(video_path)
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    # 创建输出视频文件
    output_path = "processed_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # 统计变量初始化
    frame_id = 0
    track_id_set = set()

    stframe = st.empty()  # Streamlit 实时帧显示

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        # 使用 YOLOv8 进行检测
        results = yolo_model(frame, conf=conf_thresh, iou=iou_thresh, agnostic_nms=True, imgsz=640)
        dets = []

        if results is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                if conf >= conf_thresh:
                    dets.append([x1, y1, x2, y2, conf, cls])
            dets = np.array(dets)
        else:
            dets = None

        # 更新追踪器
        res = tracker.update(dets, frame)

        for re in res:
            bbox = re[0:4]
            cls = int(re[6])
            class_name = classes[cls]
            track_id = int(re[4]) if re[4] else None

            if track_id and track_id not in track_id_set:
                track_id_set.add(track_id)
                total_count += 1
                if class_name in class_counts:
                    class_counts[class_name] += 1

        # 绘制追踪结果
        tracker.plot_results(frame, show_trajectories=True)
        out_video.write(frame)

        # 使用 Streamlit 显示
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        frame_id += 1

    # 保存并释放资源
    vid.release()
    out_video.release()

    # 显示统计信息
    st.subheader("统计结果")
    st.write(f"总果实数量: {total_count}")
    for class_name, count in class_counts.items():
        st.write(f"类别 '{class_name}' 的数量: {count}")

    # 提供视频下载链接
    with open(output_path, "rb") as f:
        st.download_button("下载处理后的视频", f, file_name="processed_output.mp4")

    # 保存统计信息到文本文件
    if save_txt_opt:
        txt_file_path = "statistics.txt"
        with open(txt_file_path, "w") as f:
            f.write(f"总果实数量: {total_count}\n")
            for class_name, count in class_counts.items():
                f.write(f"{class_name}:{count}\n")
        with open(txt_file_path, "rb") as f:
            st.download_button("下载统计结果", f, file_name="statistics.txt")
