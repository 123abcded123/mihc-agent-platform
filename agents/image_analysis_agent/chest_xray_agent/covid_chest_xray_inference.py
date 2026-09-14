"""
新冠胸片（COVID-19 Chest X-Ray）二分类推理模块。

职责：
- 基于 ImageNet 预训练的 DenseNet121，将分类头替换为 2 类输出（covid19 / normal）；
- 推理时对输入 X 光片执行 Resize(150,150) + ImageNet 均值/标准差归一化；
- 前向传播后对 logits 取 argmax，输出预测类别，并记录日志。
"""
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class ChestXRayClassification:
    # 新冠胸片二分类推理类：
    # 用途：封装 DenseNet121 二分类模型的构建、权重加载与 covid19 / normal 推理。
    def __init__(self, model_path, device=None):
        # 初始化推理器。
        # 参数:
        #     model_path: 训练好的模型权重文件路径
        #     device: 推理设备，默认按 CUDA 可用性自动选择
        # Configure logging
        # 配置日志格式，便于追踪模型加载与推理过程
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # 类别列表：下标 0 -> covid19，下标 1 -> normal
        self.class_names = ['covid19', 'normal']
        # 设备选择：优先使用指定设备，否则按 CUDA 可用性选择 cuda:0 或 CPU
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print(f"Using device: {self.device}")
        self.logger.info(f"Using device: {self.device}")
        
        # Load model
        # 构建 DenseNet121 二分类模型并加载训练好的权重
        self.model = self._build_model(weights=None)
        self._load_model_weights(model_path)
        # 模型转移到目标设备并切换为评估模式（关闭 Dropout/BatchNorm 训练行为）
        self.model.to(self.device)
        self.model.eval()
        
        # Image transformations
        # 图像预处理：Resize(150,150) + 转张量 + ImageNet 均值/标准差归一化
        self.mean_nums = [0.485, 0.456, 0.406]
        self.std_nums = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean_nums, std=self.std_nums)
        ])
    
    def _build_model(self, weights=None):
        """Initialize the DenseNet model with custom classification layer."""
        # 构建 DenseNet121 并将分类头替换为二分类线性层。
        # 参数: weights - 预训练权重（此处传入 None，即随机初始化）
        # 返回值: 改好二分类头的 DenseNet121 模型
        model = models.densenet121(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, len(self.class_names))
        return model
    
    def _load_model_weights(self, model_path):
        """Load pre-trained model weights."""
        # 从本地路径加载训练好的模型权重到当前设备。
        # 参数: model_path - 权重文件路径；加载失败时记录错误并向上抛出异常
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            # print(f"Model loaded successfully from {model_path}")
            self.logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            # print(f"Error loading model: {e}")
            # 权重加载失败：记录错误日志后重新抛出，交由上层处理
            self.logger.error(f"Error loading model: {e}")
            raise e
    
    def predict(self, img_path):
        """Predict the class of a given image."""
        # 用途：对给定 X 光片进行 covid19 / normal 二分类预测。
        # 参数: img_path - 胸片图片路径
        # 返回值: 预测类别字符串（covid19 / normal），异常时返回 None
        try:
            # 读取图片并统一转为 RGB 三通道
            image = Image.open(img_path).convert("RGB")
            # 预处理并增加 batch 维度：(C,H,W) -> (1,C,H,W)
            image_tensor = self.transform(image).unsqueeze(0)
            input_tensor = Variable(image_tensor).to(self.device)
            
            with torch.no_grad():
                # 前向传播得到 logits；argmax 取最大概率类别下标
                out = self.model(input_tensor)
                _, preds = torch.max(out, 1)
                idx = preds.cpu().numpy()[0]
                # 下标映射为类别名（covid19 / normal）
                pred_class = self.class_names[idx]
                
            # # Display Image
            # plt.imshow(np.array(image))
            # plt.title(f"Predicted: {pred_class}")
            # plt.show()

            self.logger.info(f"Predicted Class: {pred_class}")
            
            return pred_class
        except Exception as e:
            # 推理异常时记录错误并返回 None
            self.logger.error(f"Error during prediction Covid Chest X-ray: {str(e)}")
            return None

# if __name__ == "__main__":
#     classifier = ChestXRayClassification('./models/covid_chest_xray_model.pth')
#     predicted_class = classifier.predict('./images/NORMAL2-IM-0362-0001.jpeg')
#     print(f"PREDICTED CLASS: {predicted_class}")
