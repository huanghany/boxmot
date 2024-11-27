import cv2
import torch
import gdown
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from boxmot.utils import logger as LOGGER
from boxmot.appearance.reid_model_factory import (
    get_model_name,
    get_model_url,
    build_model,
    get_nr_classes,
    show_downloadable_models
)
from boxmot.utils.checks import RequirementsChecker


class BaseModelBackend:
    def __init__(self, weights, device, half):
        self.weights = weights[0] if isinstance(weights, list) else weights
        self.device = device
        self.half = half
        self.model = None
        self.cuda = torch.cuda.is_available() and self.device.type != "cpu"

        self.download_model(self.weights)
        self.model_name = get_model_name(self.weights)

        self.model = build_model(
            self.model_name,
            num_classes=get_nr_classes(self.weights),
            pretrained=not (self.weights and self.weights.is_file()),
            use_gpu=device,
        )
        self.checker = RequirementsChecker()
        self.load_model(self.weights)

    def get_crops(self, xyxys, img):
        h, w = img.shape[:2]
        # resize_dims = (128, 256)
        resize_dims = (224, 224)
        interpolation_method = cv2.INTER_LINEAR  # 选用最近邻插值
        mean_array = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std_array = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # Preallocate tensor for crops
        num_crops = len(xyxys)
        crops = torch.empty((num_crops, 3, resize_dims[1], resize_dims[0]),
                            dtype=torch.half if self.half else torch.float, device=self.device)
        combined_images = []
        for i, box in enumerate(xyxys):
            x1, y1, x2, y2 = box.astype('int')
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
            crop = img[y1:y2, x1:x2]

            # Resize and convert color in one step
            crop = cv2.resize(crop, resize_dims, interpolation=interpolation_method)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            # combined_images.append(crop)
            # Convert to tensor and normalize (convert to [0, 1] by dividing by 255 in batch later)
            crop = torch.from_numpy(crop).to(self.device, dtype=torch.half if self.half else torch.float)
            crops[i] = torch.permute(crop, (2, 0, 1))  # Change to (C, H, W)

        # Normalize the entire batch in one go
        crops = crops / 255.0

        # Standardize the batch
        crops = (crops - mean_array) / std_array
        # 使用 matplotlib 显示所有合并后的掩码图像
        # for i, img in enumerate(combined_images):
        #     plt.subplot(1, num_crops, i + 1)
        #     plt.imshow(img)
        #     plt.axis('off')
        #     plt.title(f"Crop {i + 1}")
        # plt.show()
        # plt.close()
        return crops

    def get_crops_with_mask(self, xyxys, img, masks, resize_dims=(224, 224)):  # resize_dims = (128, 256)
        """
        从图像中裁剪出目标区域，并将掩码合并到裁剪区域上。

        Args:
        - xyxys (numpy array): 包含边界框坐标的数组，形状为 (num_boxes, 4)。
        - img (numpy array): 输入图像，形状为 (H, W, 3)。
        - masks (numpy array): 掩码数组，形状为 (num_boxes, H, W)，每个框对应一个掩码。
        - resize_dims (tuple): 目标裁剪区域的尺寸 (H, W)，默认 (128, 256)。

        Returns:
        - crops (torch.Tensor): 一个形状为 (num_crops, 3, H, W) 的 Tensor，包含处理后的裁剪区域。
        """
        h, w = img.shape[:2]
        num_crops = len(xyxys)

        # 预先分配一个空的 Tensor 来保存裁剪区域
        crops = torch.empty((num_crops, 3, resize_dims[1], resize_dims[0]),
                            dtype=torch.half if self.half else torch.float, device=self.device)

        # 均值和标准差用于标准化
        mean_array = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std_array = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # 保存掩码合并后的图像列表
        combined_images = []
        for i, box in enumerate(xyxys):
            x1, y1, x2, y2 = box.astype('int')
            # 确保裁剪区域在图像范围内
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
            # 从原图像中裁剪出目标区域
            crop = img[y1:y2, x1:x2]
            # 获取对应的掩码，并调整大小到图像的尺寸
            mask_region = masks[i]  # masks 是 (num_boxes, H, W)，每个框对应一个掩码
            resized_mask = cv2.resize(mask_region, (w, h))  # 将 mask resize 为与 img 相同的大小
            mask_region = resized_mask[y1:y2, x1:x2]  # 裁剪对应区域的 mask

            # 确保掩码是 float 类型
            mask_region = mask_region.astype(np.float32)
            crop_float = crop.astype(np.float32)
            # mask_region 为 0 的地方对应区域变黑，为 1 的地方保留原图像内容
            # 生成黑色背景
            black_frame = np.zeros_like(crop)
            # 应用掩码
            mask_resized = (mask_region * 255).astype(np.uint8)  # 将掩码值调整到 [0, 255]
            colored_region = cv2.bitwise_and(crop, crop, mask=mask_resized)
            masked_crop = cv2.add(black_frame, colored_region)
            masked_crop = cv2.resize(masked_crop, resize_dims, interpolation=cv2.INTER_LINEAR)
            masked_crop = cv2.cvtColor(masked_crop, cv2.COLOR_BGR2RGB)
            # 将合并后的图像保存用于显示
            # combined_images.append(masked_crop)
            # 转换为 Tensor 并进行归一化
            masked_crop = torch.from_numpy(masked_crop).to(self.device, dtype=torch.half if self.half else torch.float)
            # 调整维度为 (C, H, W)
            crops[i] = torch.permute(masked_crop, (2, 0, 1))
        # 对整个批次进行归一化
        crops = crops / 255.0
        # 标准化处理
        crops = (crops - mean_array) / std_array

        # 使用 matplotlib 显示所有合并后的掩码图像
        # for i, img in enumerate(combined_images):
        #     plt.subplot(1, num_crops, i + 1)
        #     plt.imshow(img)
        #     plt.axis('off')
        #     plt.title(f"Crop {i + 1}")
        # plt.show()
        # plt.close()
        return crops

    @torch.no_grad()
    def get_features_mask(self, xyxys, img, masks):  # det 是 (N, 5) 格式的目标框
        if xyxys.size != 0:
            crops = self.get_crops_with_mask(xyxys, img, masks)  # 从图像中获取crop
            crops = self.inference_preprocess(crops)  # 对crop进行预处理
            features = self.forward(crops)  # 把crops放入模型中进行计算
            features = self.inference_postprocess(features)
        else:
            features = np.array([])
        features = features / np.linalg.norm(features)  # 归一化
        return features

    @torch.no_grad()
    def get_features(self, xyxys, img):  #
        if xyxys.size != 0:
            crops = self.get_crops(xyxys, img)  # 从图像中获取crop
            crops = self.inference_preprocess(crops)  # 对crop进行预处理
            features = self.forward(crops)  # 把crops放入模型中进行计算
            features = self.inference_postprocess(features)
        else:
            features = np.array([])
        features = features / np.linalg.norm(features)  # 归一化
        return features

    def warmup(self, imgsz=[(256, 128, 3)]):
        # warmup model by running inference once
        if self.device.type != "cpu":
            im = np.random.randint(0, 255, *imgsz, dtype=np.uint8)
            crops = self.get_crops(xyxys=np.array(
                [[0, 0, 64, 64], [0, 0, 128, 128]]),
                img=im
            )
            crops = self.inference_preprocess(crops)  #
            self.forward(crops)  # warmup

    def to_numpy(self, x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

    def inference_preprocess(self, x):
        if self.half:
            if isinstance(x, torch.Tensor):
                if x.dtype != torch.float16:
                    x = x.half()
            elif isinstance(x, np.ndarray):
                if x.dtype != np.float16:
                    x = x.astype(np.float16)

        if self.nhwc:
            if isinstance(x, torch.Tensor):
                x = x.permute(0, 2, 3, 1)  # Convert from NCHW to NHWC
            elif isinstance(x, np.ndarray):
                x = np.transpose(x, (0, 2, 3, 1))  # Convert from NCHW to NHWC
        return x

    def inference_postprocess(self, features):  # 推理后处理
        if isinstance(features, (list, tuple)):
            return (
                self.to_numpy(features[0]) if len(features) == 1 else [self.to_numpy(x) for x in features]
            )
        else:
            return self.to_numpy(features)

    @abstractmethod
    def forward(self, im_batch):
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def load_model(self, w):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def download_model(self, w):
        if w.suffix == ".pt":
            model_url = get_model_url(w)
            if not w.exists() and model_url is not None:
                gdown.download(model_url, str(w), quiet=False)
            elif not w.exists():
                LOGGER.error(
                    f"No URL associated with the chosen StrongSORT weights ({w}). Choose between:"
                )
                show_downloadable_models()
                exit()
