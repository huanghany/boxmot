import torch
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import platform
from streamlit_utils.utils import display_statistics_as_table, load_model


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
conf_thresh = st.sidebar.slider("置信度阈值", 0.0, 1.0, 0.1, step=0.01)
iou_thresh = st.sidebar.slider("IOU 阈值", 0.0, 1.0, 0.7, step=0.01)
save_txt_opt = st.sidebar.checkbox("保存结果到文本文件", value=False)
use_mask_opt = st.sidebar.checkbox("使用mask计算reid相似度", value=False)


yolo_model, tracker = load_model()


# 定义视频路径选项
video_paths = [
    r'/home/xplv/huanghanyang/Track_Datasets/1_艾维/20240113-103852_rack-1_left_RGB.mp4',
    r'/home/xplv/huanghanyang/Track_Datasets/1_艾维/20240113-104949_rack-5_right_RGB.mp4',
    r'/home/xplv/huanghanyang/Track_Datasets/2_工厂_phone/0726_redBerry_7_QR.mp4',
    r'/home/xplv/huanghanyang/Track_Datasets/2_工厂_phone/0804_redBerry_6.mp4',
    r'/home/xplv/huanghanyang/Track_Datasets/3_工厂_相机/0725_2.mp4',
    r'/home/xplv/huanghanyang/Track_Datasets/4_工厂_变速/2L_v20_A15.mp4',
    r'/home/xplv/huanghanyang/Track_Datasets/4_工厂_v04/strawberryVideo_20222023testDS_v040_L4_1.mp4',
    r'/home/xplv/huanghanyang/Track_Datasets/6_工厂_v04/part2_1.mp4',
    r'/home/xplv/huanghanyang/Track_Datasets/train/strawberryVideo_20222023testDS_v040_L2_2.mp4',
    r'/home/xplv/huanghanyang/Track_Datasets/bot_test/aiwei_2.mp4',
    r'/home/xplv/huanghanyang/Track_Datasets/test/aiwei_2_cut.mp4',
    r'D:\华毅\目标追踪数据集\test/aiwei_2_cut.mp4',
    r'D:\华毅\目标追踪数据集\1_艾维/20240113-104949_rack-5_right_RGB.mp4',
    r'D:\华毅\目标追踪数据集\1_艾维/20240113-104949_rack-5_right_RGB.mp4',
]

# 判断操作系统类型
system_type = platform.system()
st.sidebar.title("视频源选择")

# 根据系统类型筛选视频路径
if system_type == "Windows":
    filtered_paths = [path for path in video_paths if path.startswith('D:')]
else:
    filtered_paths = [path for path in video_paths if path.startswith('/home')]

# 视频路径选择
selected_video = st.sidebar.selectbox("选择视频文件", filtered_paths)
uploaded_video = st.sidebar.file_uploader("上传视频文件", type=["mp4", "avi", "mov"])
# 使用选定的视频路径
if selected_video:
    video_path = selected_video
    st.write(f"已选择视频路径: {video_path}")
    # 添加开始按钮
    start_button = st.sidebar.button("开始处理")

    if start_button:
        # 加载视频并处理
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
        # 显示统计结果表格
        display_statistics_as_table(class_counts)
        for class_name, count in class_counts.items():
            st.write(f"类别 '{class_name}' 的数量: {count}")

        # 提供视频下载链接
        with open(output_path, "rb") as f:
            download_button = st.download_button("下载处理后的视频", f, file_name="processed_output.mp4")

        # 处理下载
        if download_button:
            st.write("下载链接已生成，点击按钮开始下载。")
            # 注意：此时没有删除文件，下载后内容不消失
            stframe.empty()  # 如果需要清空图像显示，这一步可以去掉

