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
device = torch.device('cuda')  # 如果有 GPU，可以使用 'cuda'
yolo_model = YOLO(
    'tracking/weights/yolov8l_bestmodel_dataset3131_cls7_416_416_renamecls.pt')  # 替换为你的 YOLOv8 模型路径
yolo_model.to(device)

# 初始化追踪器
tracking_config = TRACKER_CONFIGS / 'botsort.yaml'
tracker = BotSort(
    reid_weights=Path('tracking/weights/resnet50_berry_add_6.pt'),  # ReID 模型路径
    device=0,  # 使用 CPU 进行推理
    half=False,
    track_high_thresh=0.6
)

# 批量处理文件夹中的所有视频
# video_folder = Path('/home/xplv/huanghanyang/Track_Datasets')  # 替换为视频文件夹路径
video_folder = Path(r'D:\华毅\目标追踪数据集\bad_case')  # 替换为视频文件夹路径
output_folder = Path('batch_save/test2')  # 结果保存路径
output_folder.mkdir(parents=True, exist_ok=True)

video_files = list(video_folder.glob('*.mp4'))  # 获取所有 MP4 文件

for video_path in video_files:
    print(f"处理视频: {video_path}")

    # 初始化统计数据
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

    # 打开视频文件
    vid = cv2.VideoCapture(str(video_path))

    while True:
        ret, frame = vid.read()
        if not ret:
            print("没读到 or 结束")
            break

        # YOLOv8 检测
        results = yolo_model(frame, conf=0.5, iou=0.7, agnostic_nms=True, imgsz=640,
                             classes=[[0, 1, 2, 3, 4, 6]])
        dets = []
        reid_masks = []
        if results is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                if conf >= 0.1:
                    dets.append([x1, y1, x2, y2, conf, cls])
            dets = np.array(dets)

            if results[0].masks is not None:
                for mask in results[0].masks.data.cpu().numpy():
                    reid_masks.append(mask)
                reid_masks = np.array(reid_masks)
            else:
                reid_masks = None
        else:
            dets = None
            reid_masks = None

        res = tracker.update(dets, frame)
        for re in res:
            bbox = re[0:4]
            cls = int(re[6])
            class_name = classes[int(re[6])]
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

            class_name = class_name + '_'
            line = (frame_id, class_name, track_id, int(bbox[0]), int(bbox[1]),
                    int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]), -1, -1, -1, 0)
            texts.append(("%g,%s,%g,%g,%g,%g,%g,%g,%g,%g,%g" % line))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        frame_id += 1

    vid.release()

    # 保存当前视频的统计信息
    txt_file = output_folder / f"{video_path.stem}_result.txt"
    with open(txt_file, "w") as f:
        f.writelines(text + "\n" for text in texts)
    save_statistics_to_txt(output_folder / f"{video_path.stem}_statistics.txt")
    print(f"{video_path.stem} 的统计结果保存完成")

print("所有视频处理完成。")
