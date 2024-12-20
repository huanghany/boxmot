"""
作者：黄瀚扬
时间：2024/12/16

"""
from boxmot.appearance.reid_auto_backend import ReidAutoBackend
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import os

# 配置路径
model_path = '../tracking/weights/resnet50_berry_add_6.pt'  # 使用的 ResNet 权重路径
image_folder = r'/home/xplv/huanghanyang/Track_Datasets/mask/bad_mask/aiwei_1'  # 存放图像的主文件夹
output_folder = r'./result/'
# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)
# 加载模型
model = ReidAutoBackend(weights=Path(model_path), device=0).model.model
model.eval()
device = torch.device(0)
model.to(device)

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return preprocess(img).unsqueeze(0).to(device)


def extract_features(image_tensor):
    with torch.no_grad():
        features = model(image_tensor)
    return F.normalize(features, p=2, dim=1)  # 特征归一化


def cosine_similarity(feature1, feature2):
    return torch.mm(feature1, feature2.t()).item()


# 保存单行对比图并显示相似度
def save_comparison_with_similarity(image_pairs, similarities, output_path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))  # 创建1行4列的子图

    for i, (pair, sim) in enumerate(zip(image_pairs, similarities)):
        img1 = Image.open(pair[0]).convert("RGB")
        img2 = Image.open(pair[1]).convert("RGB")

        # 显示第一张图
        axes[i * 2].imshow(img1)
        axes[i * 2].axis('off')
        axes[i * 2].set_title(f"Sim: {sim:.4f}")

        # 显示第二张图
        axes[i * 2 + 1].imshow(img2)
        axes[i * 2 + 1].axis('off')
        axes[i * 2 + 1].set_title(f"Sim: {sim:.4f}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# 批处理文件夹中的图片
def process_images_batch(image_folder, output_folder):
    files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    files.sort()  # 按名称排序文件

    # 将文件分成四个一组
    batches = [files[i:i + 4] for i in range(0, len(files), 4)]

    for idx, batch in enumerate(batches):
        if len(batch) == 4:
            img1 = os.path.join(image_folder, batch[0])
            img2 = os.path.join(image_folder, batch[1])
            img3 = os.path.join(image_folder, batch[2])
            img4 = os.path.join(image_folder, batch[3])

            # 图片对：分别对bbox和mask计算
            image_pairs = [(img1, img3), (img2, img4)]

            # 提取特征并计算相似度
            similarities = []
            for img_a, img_b in image_pairs:
                feature_a = extract_features(preprocess_image(img_a))
                feature_b = extract_features(preprocess_image(img_b))
                similarities.append(cosine_similarity(feature_a, feature_b))

            # 保存对比图
            output_path = os.path.join(output_folder, f'comparison_batch_{idx + 1}.png')
            save_comparison_with_similarity(image_pairs, similarities, output_path)
            print(f"批次 {idx + 1} 结果已保存至: {output_path}")


# 主程序执行
if __name__ == "__main__":
    process_images_batch(image_folder, output_folder)

