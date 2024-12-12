#!/bin/bash

# 输入视频文件
input_video="input_video.mp4"

# 时间片段文件
segments_file="segments.txt"

# 检查输入文件是否存在
if [[ ! -f "$input_video" || ! -f "$segments_file" ]]; then
    echo "视频文件或时间段文件不存在！"
    exit 1
fi

# 循环处理每一行
while read -r start_time duration output_name; do
    echo "正在裁剪: 开始时间=$start_time, 时长=$duration, 输出文件=$output_name"
    ffmpeg -i "$input_video" -ss "$start_time" -t "$duration" -c copy "$output_name"
done < "$segments_file"

echo "所有片段裁剪完成！"
