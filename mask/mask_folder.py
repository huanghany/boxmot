import cv2
import numpy as np
from ultralytics import YOLO
import os

# 加载 YOLO 模型
model = YOLO(r'../tracking/weights/yolov8l_bestmodel_dataset3131_cls7_416_416_renamecls.pt')

# 输入和输出目录
input_folder = r'../mask'  # 替换为你的输入目录路径
output_folder = r'save'  # 替换为你的输出目录路径
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹及其子文件夹
for root, _, files in os.walk(input_folder):
    for file in files:
        # 检查是否为图片文件
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # 构造输入文件路径
        input_path = os.path.join(root, file)

        # 构造对应的输出文件路径
        relative_path = os.path.relpath(input_path, input_folder)
        output_path = os.path.join(output_folder, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 读取输入图像
        frame = cv2.imread(input_path)
        if frame is None:
            print(f"无法读取图像：{input_path}")
            continue

        print(f"正在处理：{input_path}")

        # 使用 YOLO 模型进行预测
        results = model.predict(frame, conf=0.5, iou=0.7, imgsz=640)
        if results is None or results[0].masks is None:
            print(f"无检测结果：{input_path}")
            continue

        # 提取原始图像尺寸
        original_h, original_w = frame.shape[:2]
        output_frame = np.zeros_like(frame)  # 创建全黑背景图像

        # 提取预测结果
        confs = results[0].boxes.conf.cpu().numpy().round(2)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        masks = results[0].masks.data.cpu().numpy()

        # 遍历每个检测框
        for i, (box, mask, conf) in enumerate(zip(boxes, masks, confs)):
            x1, y1, x2, y2 = map(int, box)

            # 边界检查
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(original_w, x2), min(original_h, y2)

            # 将 mask 从输出尺寸缩放到原始图像尺寸
            resized_mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            binary_mask = (resized_mask > 0.5).astype(np.uint8)

            # 应用掩码到原图像的检测区域
            mask_resized = (binary_mask * 255).astype(np.uint8)
            colored_region = cv2.bitwise_and(frame, frame, mask=mask_resized)

            # 将结果叠加到输出图像上
            output_frame = cv2.add(output_frame, colored_region)

        # 保存结果
        cv2.imwrite(output_path, output_frame)

print(f"处理完成！结果已保存到 {output_folder}")
