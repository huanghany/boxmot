import ast
import re

input_txt = r'D:\华毅\目标追踪数据集\compare\compare_aiwei_2.txt'
output_txt = 'aiwei_2.txt'
# 用来存储三维数组
three_dimensional_array = []
# 临时数组，用来存储当前的 kl 数组
current_kl_array = []

# 文件写入
with open(output_txt, 'w', encoding='utf-8') as output_file:
    video_counter = 0  # 用来记录遇到的 "video" 数量，作为第一位的值

    # 打开输入txt文件读取（这里使用 'kl.txt' 作为输入文件名）
    with open(input_txt, 'r', encoding='utf-8') as file:
        for line in file:
            # 处理以 "kl:" 开头的行
            if line.strip().startswith("kl:"):
                # 处理 kl 行时，设置 class_type 为 1
                line = line.split("kl:")[1].strip()
                try:
                    class_type = str(1)  # 如果是 kl:，class 为 1
                    array = ast.literal_eval(line)  # 转换为 Python 列表
                    array.insert(0, class_type)
                    # current_kl_array.append(class_type)
                    current_kl_array.append(array)
                except (ValueError, SyntaxError):
                    print(f"无法解析的格式：{line}")

            # 处理以 "det:" 开头的行
            elif line.strip().startswith("det:"):
                # 处理 det 行时，设置 class_type 为 2
                line = line.split("det:")[1].strip()
                try:
                    class_type = str(2)  # 如果是 det:，class 为 2
                    array = ast.literal_eval(line)  # 转换为 Python 列表
                    array.insert(0, class_type)
                    current_kl_array.append(array)
                except (ValueError, SyntaxError):
                    print(f"无法解析的格式：{line}")

            # 处理以 "video" 开头的行
            elif line.strip().startswith("video"):
                # 每次遇到 video 时，第一位加一，写入当前的 kl 数据到文件
                for kl_array in current_kl_array:
                    # 假设数据格式为 (video_count, class, label, x, y, w, h, -1, -1, -1, 0)
                    # 其中 `kl_array` 是当前的 `kl` 数组，其他值使用默认值
                    output_data = [
                        video_counter,  # 第一位是当前视频计数
                        'Unripe_',  # 假设为固定值
                        # class_type,  # class 类型，1 或 2
                        *kl_array,  # 将 `kl_array` 中的值加入
                        -1, -1, -1,  # 默认值
                        0  # 默认值
                    ]
                    # 将数据写入文件（转换为字符串并保存一行）
                    output_file.write(','.join(map(str, output_data)) + '\n')

                # 清空 current_kl_array，准备接收下一个视频的 kl 数据
                current_kl_array = []
                # 视频计数器加一
                video_counter += 1

    # 如果文件最后有未添加的数组，确保也保存
    for kl_array in current_kl_array:
        output_data = [
            video_counter,  # 第一位是当前视频计数
            'Unripe_',  # 假设为固定值
            # class_type,  # class 类型，1 或 2
            *kl_array,  # 将 `kl_array` 中的值加入
            -1, -1, -1,  # 默认值
            0  # 默认值
        ]
        output_file.write(','.join(map(str, output_data)) + '\n')

print(f"数据已成功写入 {output_txt} 文件中。")
