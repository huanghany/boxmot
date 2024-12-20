import cv2
import os
import numpy as np
from ultralytics import YOLO
from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS


def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.
    在预测期间初始化用于对象跟踪的跟踪器
    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = TRACKER_CONFIGS / (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


# 加载 YOLO 模型
model = YOLO(r'../tracking/weights/yolov8l_bestmodel_dataset3131_cls7_416_416_renamecls.pt')

# 输入视频路径
# video_path = r'D:\华毅\目标追踪数据集\bad_case/bad_case_2.mp4'
video_path = r'/home/xplv/huanghanyang/Track_Datasets/1_艾维/20240113-104949_rack-5_right_RGB.mp4'
cap = cv2.VideoCapture(video_path)

# 创建输出文件夹
output_folder = r'/home/xplv/huanghanyang/Track_Datasets/mask/aiwei_2'
os.makedirs(output_folder, exist_ok=True)

frame_count = 0  # 帧计数器

# 逐帧处理
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"正在处理第 {frame_count} 帧...")

    # 使用 YOLO 模型进行预测
    results = model.predict(frame, conf=0.1, iou=0.7, imgsz=640)
    # model.add_callback('on_predict_start', partial(on_predict_start, persist=True))  # 添加回调函数
    # 提取原始图像尺寸
    original_h, original_w = frame.shape[:2]
    # 遍历每个检测结果
    track_id = 0
    for box, mask, conf, cls in zip(
            results[0].boxes.xyxy.cpu().numpy(),  # 检测框 [x1, y1, x2, y2]
            results[0].masks.data.cpu().numpy() if results[0].masks is not None else [None] * len(results[0].boxes),
            results[0].boxes.conf.cpu().numpy(),  # 置信度
            results[0].boxes.cls.cpu().numpy(),  # 类别
    ):
        track_id += 1
        x1, y1, x2, y2 = map(int, box)
        # 边界检查
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(original_w, x2), min(original_h, y2)
        # 去除很小的检测框
        if x2 - x1 < 50 and y2 - y1 < 50:
            continue
        # 缩放掩码到原始图像尺寸
        if mask is not None:
            resized_mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            binary_mask = (resized_mask > 0.5).astype(np.uint8)
            cropped_mask = binary_mask[y1:y2, x1:x2]

        # 裁剪检测框区域
        cropped_frame = frame[y1:y2, x1:x2]
        # 生成黑色背景
        black_frame = np.zeros_like(cropped_frame)
        # 应用掩码
        mask_resized = (cropped_mask * 255).astype(np.uint8)  # 将掩码值调整到 [0, 255]
        colored_region = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask_resized)
        final_result = cv2.add(black_frame, colored_region)
        # 保存检测框图像
        bbox_output_path = os.path.join(output_folder, f'frame_{frame_count}_id_{int(track_id)}_bbox.png')
        cv2.imwrite(bbox_output_path, cropped_frame)

        # 保存掩码图像
        if mask is not None:
            mask_output_path = os.path.join(output_folder, f'frame_{frame_count}_id_{int(track_id)}_mask.png')
            cv2.imwrite(mask_output_path, final_result)

# 释放资源
cap.release()
print(f"视频处理完成！所有检测框和掩码已保存到 {output_folder}")
