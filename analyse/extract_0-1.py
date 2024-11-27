import re

# 假设你有一个包含这些值的 txt 文件
filename = '1.txt'

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
except UnicodeDecodeError:
    print("文件读取时遇到编码错误，请检查文件的编码格式。")
