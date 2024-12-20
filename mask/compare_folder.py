"""
作者：黄瀚扬
时间：2024/12/16

"""
from torchvision.transforms import InterpolationMode

from boxmot.appearance.reid_auto_backend import ReidAutoBackend
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import os

# 配置路径
model_path = '../tracking/weights/resnet50_berry_add_1.pt'  # 使用的 ResNet 权重路径
root_folder = r'D:\华毅\目标追踪数据集\mask\bad_case'  # 存放图像的根文件夹
output_folder = r'./result_test/'
# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)
# 加载模型
model = ReidAutoBackend(weights=Path(model_path), device=0).model

device = torch.device(0)
h, w = 224, 224

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((h, w), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return preprocess(img).unsqueeze(0).to(device)

def extract_features(image_tensor):
    with torch.no_grad():
        features = model.forward(image_tensor)
    return F.normalize(features, p=2, dim=1)  # 特征归一化

def cosine_similarity(feature1, feature2):
    return torch.mm(feature1, feature2.t()).item()

# 保存所有对比结果到一张图中
def save_all_comparisons_with_similarity(images, similarities, output_path):
    num_images = len(images)
    cols = 10
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))

    # 如果只有一行子图，axes 需要转为列表
    if rows == 1:
        axes = [axes]

    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    for i, (img_path, sim) in enumerate(zip(images, similarities)):
        row, col = divmod(i, cols)
        axes[row][col].imshow(Image.open(img_path).convert("RGB"))
        axes[row][col].set_title(f"Sim: {sim:.4f}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# 处理文件夹中的相邻图片
def process_folder(image_folder, output_folder):
    files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
    files.sort()  # 按名称排序文件

    images = [os.path.join(image_folder, files[0])]
    similarities = [1.0]  # 第一张图相似度为1.0（参考点）

    for i in range(1, len(files)):
        img_a = os.path.join(image_folder, files[i - 1])
        img_b = os.path.join(image_folder, files[i])

        feature_a = extract_features(preprocess_image(img_a))
        feature_b = extract_features(preprocess_image(img_b))
        images.append(img_b)
        similarities.append(cosine_similarity(feature_a, feature_b))

    # 保存所有对比结果到一张图中
    folder_name = os.path.basename(image_folder)
    output_path = os.path.join(output_folder, f'{folder_name}_comparisons.png')
    save_all_comparisons_with_similarity(images, similarities, output_path)
    print(f"文件夹 {folder_name} 的对比结果已保存至: {output_path}")

# 批处理所有子文件夹
def process_all_folders(root_folder, output_folder):
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            process_folder(folder_path, output_folder)

# 主程序执行
if __name__ == "__main__":
    process_all_folders(root_folder, output_folder)
