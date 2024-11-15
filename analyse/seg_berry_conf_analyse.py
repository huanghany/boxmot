from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# video_path = r'D:\华毅\目标追踪数据集\1_艾维/20240113-104949_rack-5_right_RGB.mp4'
video_path = r'/home/xplv/huanghanyang/Track_Datasets/2_工厂_phone/0726_redBerry_7_QR.mp4'
# 加载YOLO模型
model = YOLO(r'../tracking/weights/yolov8l_bestmodel_dataset3131_cls7_416_416_renamecls.pt')

# 预测视频
result = model.predict(source=video_path, iou=0.7, conf=0.5, save=False, agnostic_nms=True)

# 初始化一个空的列表用于存储所有置信度
confidences_array = []

# 遍历每一帧的预测结果
for frame in result:
    # 获取当前帧的置信度
    confidences_array.extend(frame.boxes.conf.cpu().numpy())  # 将当前帧的置信度加入到数组中

# 打印所有置信度
print(confidences_array)

values_array = np.array(confidences_array)

# 计算统计量
min_val = np.min(values_array)
max_val = np.max(values_array)
mean_val = np.mean(values_array)
median_val = np.median(values_array)
std_val = np.std(values_array)

# 输出统计信息
print(f"最小值: {min_val}")
print(f"最大值: {max_val}")
print(f"均值: {mean_val}")
print(f"中位数: {median_val}")
print(f"标准差: {std_val}")

# 设置中文字体
# font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑字体路径（Windows）
font_path = "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"  #
# 如果你在 Mac 或 Linux 上，路径可能不同，具体可以选择合适的字体路径
my_font = font_manager.FontProperties(fname=font_path)

# 绘制直方图
counts, bins, patches = plt.hist(values_array, bins=10, edgecolor='black', alpha=0.7)
# 在每个柱子上方显示数量
for count, patch in zip(counts, patches):
    # 获取柱子的坐标
    height = patch.get_height()
    # 显示文本，位置为柱子的中间顶部
    plt.text(patch.get_x() + patch.get_width() / 2, height + 0.01, f'{int(count)}',
             ha='center', va='bottom', fontsize=10, fontproperties=my_font)
plt.title('Conf距离分布情况', fontproperties=my_font)  # IOU距离分布情况 ReID距离分布情况
plt.xlabel('值', fontproperties=my_font)
plt.ylabel('频率', fontproperties=my_font)

# 显示统计信息
plt.figtext(0.45, 0.8, f'Min: {min_val:.5f}', fontsize=10)
plt.figtext(0.45, 0.75, f'Max: {max_val:.5f}', fontsize=10)
plt.figtext(0.45, 0.7, f'Mean: {mean_val:.5f}', fontsize=10)
plt.figtext(0.45, 0.65, f'Median: {median_val:.5f}', fontsize=10)
plt.figtext(0.45, 0.6, f'Stdev: {std_val:.5f}', fontsize=10)

# 显示图形

plt.savefig('factory_Conf.png', dpi=300, bbox_inches='tight')  # IOU_dists ReID_dists
# plt.show()
plt.close()
