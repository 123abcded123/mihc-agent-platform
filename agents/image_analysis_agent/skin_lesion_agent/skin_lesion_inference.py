"""
皮肤病灶（Skin Lesion）分割推理模块。

职责：
- 定义经典 U-Net 分割网络：编码器逐层下采样至 1024 通道，解码器上采样并与编码器特征做跳跃连接（skip connection）；
- 从 Google Drive 下载训练好的权重并加载模型；
- 对输入皮肤病灶图片执行分割，将预测掩码以 alpha=0.4 叠加到原图上，保存 segmentation_plot.png 供展示。
"""
import os
import cv2
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from .model_download import download_model_checkpoint

# Configure logging
# 配置日志格式，便于追踪模型下载、加载与分割过程
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Device setup
# 设备选择：优先使用 CUDA，否则退回 CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

class UNet(nn.Module):
    """U-Net model for image segmentation."""
    # 图像分割 U-Net 模型：
    # 用途：编码器逐层下采样提取特征（通道数 3->64->128->256->512->1024），
    #       解码器通过转置卷积上采样恢复分辨率，并与编码器特征做跳跃连接实现精确定位。
    def __init__(self, n_channels, n_classes):
        # 初始化 U-Net 网络结构。
        # 参数:
        #     n_channels: 输入图像通道数（RGB 为 3）
        #     n_classes: 输出分割类别数（单类病灶掩码为 1）
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Contracting path (encoder)
        # 编码器（收缩路径）：卷积 + 最大池化逐级下采样，通道数翻倍至 1024
        self.conv1 = nn.Conv2d(self.n_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Expansive path (decoder)
        # 解码器（扩张路径）：转置卷积上采样 + 卷积融合，逐级恢复分辨率
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(64, self.n_classes, kernel_size=1)

    def forward(self, x):
        """Forward pass of U-Net."""
        # U-Net 前向传播。
        # 参数: x - 输入图像张量
        # 返回值: 与输入分辨率一致的分割 logits 张量
        # 编码器：逐级卷积 + 池化下采样，保存各级特征用于跳跃连接
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(self.pool(x1)))
        x3 = F.relu(self.conv3(self.pool(x2)))
        x4 = F.relu(self.conv4(self.pool(x3)))
        x5 = F.relu(self.conv5(self.pool(x4)))

        # 解码器：上采样后与编码器同级特征拼接，恢复细节
        x6 = F.relu(self.upconv1(x5))
        # 跳跃连接：concat 编码器特征实现精确定位
        x6 = torch.cat([x4, x6], dim=1)
        x6 = F.relu(self.conv6(x6))
        x7 = F.relu(self.upconv2(x6))
        # 跳跃连接：concat 编码器特征实现精确定位
        x7 = torch.cat([x3, x7], dim=1)
        x7 = F.relu(self.conv7(x7))
        x8 = F.relu(self.upconv3(x7))
        # 跳跃连接：concat 编码器特征实现精确定位
        x8 = torch.cat([x2, x8], dim=1)
        x8 = F.relu(self.conv8(x8))
        x9 = F.relu(self.upconv4(x8))
        # 跳跃连接：concat 编码器特征实现精确定位
        x9 = torch.cat([x1, x9], dim=1)
        x9 = F.relu(self.conv9(x9))
        # 1x1 卷积将通道数压缩到类别数，输出最终 logits
        x10 = self.conv10(x9)

        return x10


class SkinLesionSegmentation:
    """Handles skin lesion segmentation using a trained U-Net model."""
    # 皮肤病灶分割类：
    # 用途：加载训练好的 U-Net 模型，对皮肤病灶图片执行分割并生成掩码叠加图。
    
    def __init__(self, model_path):
        # 初始化分割器。
        # 参数: model_path - U-Net 权重文件保存路径（不存在时自动从 Google Drive 下载）
        self.model_path = model_path
        self.device = DEVICE
        # 加载 U-Net 模型
        self.model = self._load_model()

    def _load_model(self):
        """Load the trained U-Net model."""
        # 加载训练好的 U-Net 模型。
        # 返回值: 已加载权重并处于评估模式的 U-Net 模型
        try:
            # with safe_globals([UNet]):
            #     model = torch.load(self.model_path, weights_only=False, map_location=self.device)
            # Call this before using the model
            # 使用模型前先确保权重文件存在：不存在则用 gdown 从 Google Drive 下载
            download_model_checkpoint('1rvn4ucOH6UBoNk-GB9bUWuGTLkNIVUf0', self.model_path)
            model = UNet(n_channels=3, n_classes=1).to(self.device)  # Explicitly initialize UNet
            # model.load_state_dict(torch.load(self.model_path, weights_only=False, map_location=self.device), strict=False)
            # 加载权重文件中的 state_dict 到 U-Net 模型
            model.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device))['state_dict'])
            # model = torch.load(self.model_path, map_location=torch.device(DEVICE))
            # 切换到评估模式
            model.eval()
            logger.info(f"Model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            # 模型加载失败：记录错误日志后重新抛出
            logger.error(f"Error loading model: {e}")
            raise e

    def _overlay_mask(self, img, mask, output_path):
        """Overlay the segmentation mask on the original image."""
        # 将分割掩码叠加到原图上并保存。
        # 参数:
        #     img: 归一化后的原图（H,W,3）
        #     mask: 预测掩码（H,W）
        #     output_path: 叠加图保存路径
        # 返回值: 保存成功返回 True，异常时抛出
        try:
            # 单通道掩码复制成三通道，便于与彩色原图叠加
            mask_stacked = np.stack((mask,) * 3, axis=-1)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.axis("off")
            # 先画原图，再以 alpha=0.4 半透明叠加掩码
            ax.imshow(img)
            ax.imshow(mask_stacked, alpha=0.4)
            # plt.savefig("overlayed_plot.png", bbox_inches="tight")
            # 保存叠加结果图（紧凑裁剪）
            plt.savefig(output_path, bbox_inches="tight")
            logger.info("Overlayed segmentation mask saved as 'overlayed_plot.png'")
            # return "overlayed_plot.png"
            return True
        except Exception as e:
            # 叠加图生成失败：记录错误并重新抛出
            logger.error(f"Error generating overlay: {e}")
            raise e
    
    def predict(self, image_path, output_path):
        """Segment lesion in an image and return overlaid visualization."""
        # 用途：对皮肤病灶图片执行 U-Net 分割，并将掩码叠加图保存到输出路径。
        # 参数:
        #     image_path: 皮肤病灶图片路径
        #     output_path: 叠加结果图保存路径
        # 返回值: 叠加图生成结果（True / 异常抛出）
        try:
            # 读取图片并做 BGR->RGB 转换及 [0,1] 归一化
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0,1]
            # 调整到模型输入尺寸 256x256
            img_resized = cv2.resize(img, (256, 256))
            # 转张量并调整维度：(H,W,C) -> (1,C,H,W)，移动到设备
            img_tensor = torch.Tensor(img_resized).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)

            with torch.no_grad():
                # 前向传播得到预测掩码，去掉 batch/通道维度
                generated_mask = self.model(img_tensor).squeeze().cpu().numpy()

            # Resize mask to match original image dimensions
            # 将掩码缩放回原图尺寸，以便与原图对齐叠加
            generated_mask_resized = cv2.resize(generated_mask, (img.shape[1], img.shape[0]))
            return self._overlay_mask(img, generated_mask_resized, output_path)

        except Exception as e:
            # 分割过程异常：记录错误并重新抛出
            logger.error(f"Error during segmentation: {e}")
            raise e


# # Example Usage
# if __name__ == "__main__":
#     segmenter = SkinLesionSegmentation(model_path="./models/skin_lesion_segmentation.pth")
#     segmented_image = segmenter.predict("./images/ISIC_0020840.jpg", "./segmentation_plot.png")
#     logger.info(f"Segmentation completed. Output saved at: {segmented_image}")
