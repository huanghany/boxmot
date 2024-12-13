# ffmpeg -i input.mp4 -c:v libx264 -preset slow -crf 23 -c:a aac -b:a 128k output.mp4


import os
import subprocess

# 输入文件夹路径
input_folder = "../save/20241012"
# 输出文件夹路径
output_folder = "../save/1"

# 支持的视频扩展名
video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有文件
for root, _, files in os.walk(input_folder):
    for file in files:
        # 检查文件扩展名
        ext = os.path.splitext(file)[1].lower()
        if ext in video_extensions:
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_folder, os.path.splitext(file)[0] + ".mp4")

            # FFmpeg 转换命令
            command = [
                "ffmpeg",
                "-i", input_path,  # 输入文件
                "-c:v", "libx264",  # 视频编码格式 H.264
                "-preset", "slow",  # 编码速度
                "-crf", "23",  # 压缩质量，0~51，数值越低质量越高
                "-c:a", "aac",  # 音频编码格式
                "-b:a", "128k",  # 音频比特率
                output_path
            ]

            print(f"正在处理: {input_path}")
            subprocess.run(command)
            print(f"已保存到: {output_path}")
