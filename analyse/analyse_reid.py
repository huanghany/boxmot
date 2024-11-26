import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import rcParams
import re

# rcParams['font.sans-serif'] = ['arphic']


# 假设你有一个包含这些值的 txt 文件
filename = '../log/aiwei_2_reid_dist_224_log.txt'  # log_iou_1 log_emb_1

# 读取文件并提取小于1的数值
try:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用正则表达式提取所有小于1的浮动值
    matches = re.findall(r'[+-]?\d*\.\d+', content)  # 匹配浮动数值

    # 转换为浮动数值并保存到数组中
    values = [float(match) for match in matches]

    # 输出结果
    print("提取的值:", values)

    # 转换为 NumPy 数组方便后续计算
    values_array = np.array(values)

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
    font_path = "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"  # 微软雅黑字体路径（Windows）
    # # 如果你在 Mac 或 Linux 上，路径可能不同，具体可以选择合适的字体路径
    my_font = font_manager.FontProperties(fname=font_path)

    # 绘制直方图
    counts, bins, patches = plt.hist(values_array, bins=10, edgecolor='black', alpha=0.7)
    # 在每个柱子上方显示数量
    for count, patch in zip(counts, patches):
        # 获取柱子的坐标
        height = patch.get_height()
        # 显示文本，位置为柱子的中间顶部
        plt.text(patch.get_x() + patch.get_width() / 2, height + 0.01, f'{int(count)}',
                 ha='center', va='bottom', fontsize=10)
    plt.title('ReID_dist', font=my_font)  # IOU距离分布情况 ReID距离分布情况
    plt.xlabel('值')
    plt.ylabel('频率')  # , fontproperties=my_font

    # 显示统计信息
    plt.figtext(0.65, 0.8, f'Min: {min_val:.5f}', fontsize=10)
    plt.figtext(0.65, 0.75, f'Max: {max_val:.5f}', fontsize=10)
    plt.figtext(0.65, 0.7, f'Mean: {mean_val:.5f}', fontsize=10)
    plt.figtext(0.65, 0.65, f'Median: {median_val:.5f}', fontsize=10)
    plt.figtext(0.65, 0.6, f'Stdev: {std_val:.5f}', fontsize=10)

    # 显示图形

    plt.savefig('aiwei_2_ReID_dists_224.png', dpi=300, bbox_inches='tight')  # IOU_dists ReID_dists
    # plt.show()
    plt.close()
except UnicodeDecodeError:
    print("文件读取时遇到编码错误，请检查文件的编码格式。")


