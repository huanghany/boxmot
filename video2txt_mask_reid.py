import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import BotSort
from boxmot import StrongSort
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


# Load YOLOv8 model
device = torch.device('cuda')  # Use 'cuda' if you have a GPU
yolo_model = YOLO(
    'tracking/weights/yolov8l_bestmodel_dataset3131_cls7_416_416_renamecls.pt')  # Replace with your YOLOv8 model path if necessary
yolo_model.to(device)

# Initialize the tracker
# tracking_config = TRACKER_CONFIGS / 'botsort.yaml'
# tracker = BotSort(
#     reid_weights=Path('tracking/weights/resnet50_berry_add_6.pt'),  # Path to ReID model
#     device=0,  # Use CPU for inference
#     half=False,
#     track_high_thresh=0.6
# )
tracking_config = TRACKER_CONFIGS / 'strongsort.yaml'
tracker = StrongSort(
    reid_weights=Path('tracking/weights/resnet50_berry_add_6.pt'),  # Path to ReID model
    device=0,  # Use CPU for inference
    half=False,
    max_cos_dist=0.4
)

# Open the video file
# video_path = r'/home/xplv/huanghanyang/Track_Datasets/1_艾维/20240113-104949_rack-5_right_RGB.mp4'
video_path = r'/home/xplv/huanghanyang/Track_Datasets/1_艾维/20240113-103852_rack-1_left_RGB.mp4'
# video_path = r'/home/xplv/huanghanyang/Track_Datasets/6_工厂_v04/part2_1.mp4'
# video_path = r'D:\华毅\目标追踪数据集\1_艾维/20240113-104949_rack-5_right_RGB.mp4'
vid = cv2.VideoCapture(video_path)
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
    # Capture frame-by-frame
    ret, frame = vid.read()
    # If ret is False, it means we have reached the end of the video
    if not ret:
        print("没读到 or 结束")
        break

    # Perform detection with YOLOv8
    results = yolo_model(frame, conf=0.5, iou=0.7, agnostic_nms=True, imgsz=640,
                         classes=[[0, 1, 2, 3, 4, 6]])

    # Convert detections to numpy array (N X (x, y, x, y, conf, cls))
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
    # 是否用mask掩码做为reid相似度计算标准
    mask_reid = False  # False True
    if mask_reid:
        res = tracker.update(dets, frame, reid_masks)
    else:
        res = tracker.update(dets, frame)  # dets:-->(x, y, x, y, id, conf, cls, ind)
    print("track result: ", res)

    for re in res:
        bbox = re[0:4]  # 从张量转换为列表
        cls = int(re[6])  # 类别
        class_name = classes[int(re[6])]  # 获取类别名
        # mask =
        if re[4] is None:
            continue
        track_id = int(re[4])

        if track_id not in track_id_set:
            track_id_set.add(track_id)  # 将track_id加入集合
            total_count += 1  # 更新总数量

            # 更新每个类别的数量
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1
        class_name = class_name + '_'
        line = (frame_id, class_name, track_id, int(bbox[0]), int(bbox[1]),
                int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]), -1, -1, -1, 0)
        print(line)
        texts.append(("%g,%s,%g,%g,%g,%g,%g,%g,%g,%g,%g" % line))
    # Plot tracking results on the image
    # tracker.plot_results(frame, show_trajectories=True)

    # cv2.imshow('BoXMOT + YOLOv8', frame)

    # Simulate wait for key press to continue, press 'q' to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    frame_id += 1
# Release resources
vid.release()
cv2.destroyAllWindows()

save_txt_opt = False  # 是否保存txt

if texts and save_txt_opt:
    Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # 创建目录
    with open(txt_file, "w") as f:
        f.writelines(text + "\n" for text in texts)

print_fruit_statistics()
# source_path = Path('save')
# source_dir = source_path.parent
# source_name = source_path.stem
# result_file = source_dir / f"{source_name}_result_bot_berry_change_test2.txt"
result_file = "save/result.txt"
if save_txt_opt:
    save_statistics_to_txt(result_file)
    print(f"结果已保存至{result_file}")
