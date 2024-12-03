import streamlit as st
import platform
import cv2
import numpy as np
from pathlib import Path
from streamlit_utils.utils import display_statistics_as_table, load_model, save_tracking_results, save_statistics

# 全局变量定义
classes = ['Ripe', 'Ripe7', 'Ripe4', 'Ripe2', 'Unripe', 'Flower', 'Disease']
class_counts = {cls: 0 for cls in classes}
total_count = 0

# 根据操作系统选择路径列表
is_windows = platform.system() == "Windows"
video_paths = [
    r'D:\华毅\目标追踪数据集\test/aiwei_2_cut.mp4',
    r'D:\华毅\目标追踪数据集\1_艾维/20240113-104949_rack-5_right_RGB.mp4',
    r'D:\华毅\目标追踪数据集\train/strawberryVideo_20222023testDS_v040_L2_2.mp4',
    r'D:\华毅\目标追踪数据集\combine/combine_1.mp4'
] if is_windows else [
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
    # 可添加更多 Linux 路径
]


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
            reid_masks = np.array([mask for mask in results[0].masks.data.cpu().numpy()])

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


# Streamlit 界面
def main():
    st.title("目标追踪数据分析")
    st.sidebar.title("参数设置")
    # 路径和参数选择
    # 视频路径选择方式：预设或自定义
    path_option = st.radio(
        "视频路径来源",
        options=["选择预设路径", "自定义输入路径"],
        help="选择预设路径或手动输入视频路径"
    )
    # 根据选择显示对应输入框
    if path_option == "选择预设路径":
        video_path = st.selectbox("选择视频路径", video_paths)
    else:
        video_path = st.text_input("输入视频路径", help="请输入视频文件的完整路径")
    tracker_type = st.selectbox("选择追踪器类型", ["BotSort", "StrongSort", "DeepSORT", "OC-SORT"])
    conf_thresh = st.sidebar.slider("置信度阈值", 0.0, 1.0, 0.1, step=0.01)
    iou_thresh = st.sidebar.slider("IOU 阈值", 0.0, 1.0, 0.7, step=0.01)
    save_txt_opt = st.sidebar.checkbox("保存结果到文本文件", value=False)
    use_mask_opt = st.sidebar.checkbox("使用mask计算reid相似度", value=False)

    # 保存路径设置
    save_path_type = st.sidebar.radio("选择保存路径类型", ["保存到视频来源相同目录", "自定义目录"])
    if save_path_type == "保存到视频来源相同目录":
        save_dir = Path(video_path).parent  # 获取视频来源目录
    else:
        save_dir = st.sidebar.text_input("输入自定义保存目录", value="save")

    # 自定义文件名后缀
    custom_suffix = st.sidebar.text_input("计数结果文件后缀", value="_result.txt")
    video_name = Path(video_path).stem  # 提取视频文件名（不含扩展名）
    save_file_name = f"{video_name}{custom_suffix}"  # 动态生成文件名
    # 添加追踪结果文件后缀选项
    tracking_suffix = st.sidebar.text_input("追踪结果文件后缀", value="_tracking.txt",
                                            help="自定义追踪结果文件的后缀，例如 '_tracking.txt'")

    if video_path:
        st.write(f"已选择视频路径: {video_path}")
        # 添加开始按钮
        start_button = st.button("开始处理")
        # 加载模型
        yolo_model, tracker = load_model(tracker_type)
        if start_button:
            process_video(
                video_path=video_path,
                yolo_model=yolo_model,
                tracker=tracker,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                use_mask_opt=use_mask_opt,
                save_dir=save_dir,
                save_file_name=save_file_name,
                tracking_suffix=tracking_suffix,
                save_txt_opt=save_txt_opt
            )


def process_video(video_path, yolo_model, tracker, conf_thresh, iou_thresh, use_mask_opt, save_dir, save_file_name,
                  tracking_suffix, save_txt_opt):
    """
    处理视频并统计追踪结果。
    """
    global total_count, class_counts
    vid = cv2.VideoCapture(video_path)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    output_path = "processed_output.mp4"
    out_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    track_id_set = set()


    # 保存追踪结果相关变量
    texts = []  # 用于存储追踪结果的每一行
    tracking_file_name = f"{Path(video_path).stem}{tracking_suffix}"  # 自动根据后缀生成文件名

    st.write(f"视频总帧数: {total_frames}")
    stframe, stframetxt = st.empty(), st.empty()

    frame_id = 0
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        stframetxt.text(f"正在处理帧 {frame_id}")

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

    # 显示统计信息
    st.write(line)
    st.subheader("统计结果")
    st.write(f"总果实数量: {total_count}")
    display_statistics_as_table(class_counts)

    # 保存追踪结果
    if save_txt_opt:
        # 统计文件路径
        statistics_path = save_statistics(save_dir, save_file_name, total_count, class_counts)
        st.write(f"统计结果已保存至 {statistics_path}")
        # 更新动态生成追踪结果文件名

        # 追踪结果文件路径
        tracking_path = save_tracking_results(save_dir, tracking_file_name, texts)
        st.write(f"追踪结果已保存至 {tracking_path}")

    # 提供视频下载链接
    with open(output_path, "rb") as f:
        st.download_button("下载处理后的视频", f, file_name="result.mp4")


# 程序入口
if __name__ == "__main__":
    main()
