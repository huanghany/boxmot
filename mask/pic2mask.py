import cv2
import numpy as np
from ultralytics import YOLO  # 假设使用 YOLOv8

# 加载 YOLO 模型
model = YOLO(r'../tracking/weights/yolov8l_bestmodel_dataset3131_cls7_416_416_renamecls.pt')
# result = model.predict(source=r'/home/xplv/huanghanyang/Track_Datasets/bot_test/aiwei_2.mp4',
#                        iou=0.7, conf=0.5, save=True, agnostic_nms=True)
# 读取输入图像
image = cv2.imread('test.jpg')

# 使用 YOLO 模型进行预测
results = model.predict(image, conf=0.1, iou=0.7, imgsz=640)
# 提取原始图像尺寸
original_h, original_w = image.shape[:2]

# 提取预测结果
boxes = results[0].boxes.xyxy.cpu().numpy()  # 检测框 [x1, y1, x2, y2]
masks = results[0].masks.data.cpu().numpy()  # 分割掩码，大小为 384x640（假设）

# YOLO 输出掩码的尺寸
mask_h, mask_w = masks.shape[1:]

# 遍历每个检测框
for i, (box, mask) in enumerate(zip(boxes, masks)):
    x1, y1, x2, y2 = map(int, box)

    # 边界检查
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(original_w, x2), min(original_h, y2)

    # 将 mask 从输出尺寸缩放到原始图像尺寸
    resized_mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    binary_mask = (resized_mask > 0.5).astype(np.uint8)  # 二值化掩码

    # 从全尺寸掩码中裁剪出检测框部分
    cropped_mask = binary_mask[y1:y2, x1:x2]

    # 裁剪检测框区域
    cropped_frame = image[y1:y2, x1:x2]

    # 生成黑色背景
    black_frame = np.zeros_like(cropped_frame)

    # 应用掩码
    mask_resized = (cropped_mask * 255).astype(np.uint8)  # 将掩码值调整到 [0, 255]
    colored_region = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask_resized)
    final_result = cv2.add(black_frame, colored_region)
    # 保存结果
    output_path = f'save_1/detection_{i + 1}.png'  # 指定输出路径
    cv2.imwrite(output_path, final_result)


print("所有检测框已处理并保存！")
