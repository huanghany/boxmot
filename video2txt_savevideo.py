# 导入必要的库
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import BotSort
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS


def print_fruit_statistics():
    global total_count, class_counts
    print(f"总果实数量: {total_count}")
    for class_name, count in class_counts.items():
        print(f"类别 '{class_name}' 的数量: {count}")


def save_statistics_to_txt(txt_file):
    """保存统计信息到txt文件"""
    with open(txt_file, "w") as f:
        f.write(f"总果实数量: {total_count}\n")
        for class_name, count in class_counts.items():
            f.write(f"{class_name}:{count}\n")


# 加载 YOLOv8 模型
device = torch.device('cuda')  # 使用 GPU
yolo_model = YOLO('tracking/weights/yolov8l_bestmodel_dataset3131_cls7_416_416_renamecls.pt')  # 替换为你的模型路径
yolo_model.to(device)
# 初始化追踪器
tracking_config = TRACKER_CONFIGS / 'botsort.yaml'
tracker = BotSort(
    reid_weights=Path('tracking/weights/resnet50_berry_add_6.pt'),  # ReID 模型路径
    device=0,  # 使用 GPU
    half=False,
    track_high_thresh=0.6
)

# 打开视频文件
video_path = r'D:\华毅\目标追踪数据集\1_艾维/20240113-104949_rack-5_right_RGB.mp4'
vid = cv2.VideoCapture(video_path)

# 获取视频宽高和帧率
frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vid.get(cv2.CAP_PROP_FPS))

# 定义保存视频的编码器和输出路径
output_path = 'save/aiwei_2.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 格式
out_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_id = 0
track_id_set = set()
total_count = 0  # 总果实数量
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
texts = []
txt_file = 'save/test.txt'

while True:
    # 捕获每帧
    ret, frame = vid.read()
    if not ret:  # 视频结束
        break

    # 使用 YOLOv8 进行检测
    results = yolo_model(frame, conf=0.1, iou=0.7, agnostic_nms=True, imgsz=640, classes=[[0, 1, 2, 3, 4, 6]])

    # 将检测结果转换为 numpy 数组
    if results is not None:
        dets = []
        for box in results[0].boxes:  # 遍历检测框
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 框坐标
            conf = box.conf[0].cpu().numpy()  # 置信度
            cls = box.cls[0].cpu().numpy()  # 类别
            if conf >= 0.1:  # 置信度阈值
                dets.append([x1, y1, x2, y2, conf, cls])
        dets = np.array(dets)
    else:
        dets = None

    # 更新追踪器
    res = tracker.update(dets, frame)  # --> M X (x, y, x, y, id, conf, cls, ind)

    for re in res:
        bbox = re[0:4]
        cls = int(re[6])  # 类别
        class_name = classes[int(re[6])]  # 类别名

        if re[4] is None:
            continue
        track_id = int(re[4])

        if track_id not in track_id_set:
            track_id_set.add(track_id)
            total_count += 1

            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1

    # 绘制追踪结果
    tracker.plot_results(frame, show_trajectories=True)

    # 显示结果
    cv2.imshow('BoXMOT + YOLOv8', frame)

    # 保存帧到视频文件
    out_video.write(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    frame_id += 1

# 释放资源
vid.release()
out_video.release()  # 关闭视频写入
cv2.destroyAllWindows()

save_txt_opt = False  # 是否保存txt

# 保存统计信息
if texts and save_txt_opt:
    Path(txt_file).parent.mkdir(parents=True, exist_ok=True)
    with open(txt_file, "w") as f:
        f.writelines(text + "\n" for text in texts)

print_fruit_statistics()
result_file = "save/result.txt"
if save_txt_opt:
    save_statistics_to_txt(result_file)
    print(f"结果已保存至{result_file}")
