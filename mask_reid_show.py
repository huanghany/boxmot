import streamlit as st
import platform
import cv2
import numpy as np
from pathlib import Path
from streamlit_utils.utils import display_statistics_as_table, load_model

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

# 根据操作系统选择路径列表
is_windows = platform.system() == "Windows"

# 定义视频路径选项
linux_video_paths = [
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
    r'/home/xplv/huanghanyang/Track_Datasets/aiwei_analyse/rack-2_right_RGB.mp4',
    r'/home/xplv/huanghanyang/Track_Datasets/combine/combine_1.mp4',
]
windows_video_paths = [
    r'D:\华毅\目标追踪数据集\test/aiwei_2_cut.mp4',
    r'D:\华毅\目标追踪数据集\1_艾维/20240113-104949_rack-5_right_RGB.mp4',
    r'D:\华毅\目标追踪数据集\train/strawberryVideo_20222023testDS_v040_L2_2.mp4',
]

video_paths = windows_video_paths if is_windows else linux_video_paths

# Streamlit 界面
st.title("目标追踪数据分析")

# 添加路径选择器和开始按钮
video_path = st.selectbox("选择视频路径", video_paths)

# 添加追踪器选择
tracker_options = ["BotSort", "StrongSORT", "DeepSORT", "OC-SORT"]
tracker_type = st.selectbox("选择追踪器类型", tracker_options)

conf_thresh = st.sidebar.slider("置信度阈值", 0.0, 1.0, 0.1, step=0.01)
iou_thresh = st.sidebar.slider("IOU 阈值", 0.0, 1.0, 0.7, step=0.01)
save_txt_opt = st.sidebar.checkbox("保存结果到文本文件", value=False)
use_mask_opt = st.sidebar.checkbox("使用mask计算reid相似度", value=False)

yolo_model, tracker = load_model()
# 设置保存选项
txt_file = st.sidebar.text_input("保存路径", value="save/statistics.txt")

if video_path:
    video_path = video_path
    st.write(f"已选择视频路径: {video_path}")
    # 添加开始按钮
    start_button = st.button("开始处理")

    if start_button:
        # 加载视频并处理
        vid = cv2.VideoCapture(video_path)
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        st.write(f"视频总帧数:{total_frames}")
        # 创建输出视频文件
        output_path = "processed_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # 统计变量初始化
        frame_id = 0
        track_id_set = set()
        texts = []
        stframe = st.empty()  # Streamlit 实时帧显示
        stframetxt = st.empty()
        stframeinf = st.empty()
        while True:
            ret, frame = vid.read()
            if not ret:
                break
            stframetxt.text(f"正在处理帧 {frame_id}")  # 在占位符中显示当前帧的信息
            # 使用 YOLOv8 进行检测
            results = yolo_model(frame, conf=conf_thresh, iou=iou_thresh, agnostic_nms=True, imgsz=640,
                                 classes=[[0, 1, 2, 3, 4, 6]])
            dets = []

            if results is not None:
                dets = []
                reid_masks = []
                for box in results[0].boxes:  # Iterate through each detected box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
                    conf = box.conf[0].cpu().numpy()  # Confidence score
                    cls = box.cls[0].cpu().numpy()  # Class label
                    if conf >= 0.1:  # Confidence threshold
                        dets.append([x1, y1, x2, y2, conf, cls])
                dets = np.array(dets)

                if results[0].masks is not None:  # 检查是否有 mask
                    for mask in results[0].masks.data.cpu().numpy():  # 从 masks.data 提取 NumPy 数组
                        reid_masks.append(mask)  # 添加到 reid_masks 列表中
                    reid_masks = np.array(reid_masks)  # 转换为 NumPy 数组
                else:
                    reid_masks = None
            else:
                dets = None
                reid_masks = None

            # 更新追踪器
            if use_mask_opt:
                res = tracker.update(dets, frame, reid_masks)
            else:
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
                class_name = class_name + '_'
                line = (frame_id, class_name, track_id, int(bbox[0]), int(bbox[1]),
                        int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]), -1, -1, -1, 0)

                texts.append(("%g,%s,%g,%g,%g,%g,%g,%g,%g,%g,%g" % line))
            # 绘制追踪结果
            tracker.plot_results(frame, show_trajectories=True)
            out_video.write(frame)

            # 使用 Streamlit 显示
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            frame_id += 1

        # 保存并释放资源
        vid.release()
        out_video.release()
        stframeinf = st.text(line)
        # 显示统计信息
        st.subheader("统计结果")
        st.write(f"总果实数量: {total_count}")
        # 显示统计结果表格
        display_statistics_as_table(class_counts)
        for class_name, count in class_counts.items():
            st.write(f"类别 '{class_name}' 的数量: {count}")

        # 保存统计结果
        if save_txt_opt:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)
            with open(txt_file, "w") as f:
                f.write(f"总果实数量: {total_count}\n")
                for class_name, count in class_counts.items():
                    f.write(f"{class_name}: {count}\n")
            st.write(f"统计结果已保存至 {txt_file}")

        # if texts and save_txt_opt:
        #     Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # 创建目录
        #     with open(txt_file, "w") as f:
        #         f.writelines(text + "\n" for text in texts)

        # 提供视频下载链接
        with open(output_path, "rb") as f:
            download_button = st.download_button("下载处理后的视频", f, file_name="processed_output.mp4")

        # 处理下载
        if download_button:
            st.write("下载链接已生成，点击按钮开始下载。")
            # 注意：此时没有删除文件，下载后内容不消失
            stframe.empty()  # 如果需要清空图像显示，这一步可以去掉

