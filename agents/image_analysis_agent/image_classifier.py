"""
医学图片类型分类模块。

职责：
- 将本地图片读取为 base64 dataURL，随提示词发送给 GPT-4o 视觉模型；
- 由视觉模型判定图片是否为医学影像，并分类为
  'BRAIN MRI SCAN' / 'CHEST X-RAY' / 'SKIN LESION' / 'OTHER' / 'NON-MEDICAL'；
- 返回 JSON 分类结果（image_type、reasoning、confidence），供上游路由到对应子 Agent。
"""
import os
import json
import base64
from mimetypes import guess_type

from typing import TypedDict
from langchain_core.output_parsers import JsonOutputParser

class ClassificationDecision(TypedDict):
    """Output structure for the decision agent."""
    # 视觉模型输出 JSON 的结构定义：图片类型、推理过程、置信度
    image_type: str
    reasoning: str
    confidence: float

class ImageClassifier:
    """Uses GPT-4o Vision to analyze images and determine their type."""
    # 图片类型分类器：调用 GPT-4o 视觉模型分析图片并判定其类型
    
    def __init__(self, vision_model):
        """
        初始化分类器。

        参数:
            vision_model: 支持视觉输入的 LLM 实例（如 GPT-4o）
        """
        # 保存视觉模型引用，供分类请求调用
        self.vision_model = vision_model
        # 初始化 JSON 输出解析器：将模型返回内容解析为 ClassificationDecision 结构
        self.json_parser = JsonOutputParser(pydantic_object=ClassificationDecision)
        
    def local_image_to_data_url(self, image_path: str) -> str:
        """
        Get the url of a local image
        """
        # 将本地图片转换为 base64 dataURL，供视觉模型直接读取图片内容。
        # 参数: image_path - 本地图片路径
        # 返回值: 形如 data:<MIME类型>;base64,<编码内容> 的字符串
        # 根据文件扩展名猜测 MIME 类型（如 image/jpeg、image/png）
        mime_type, _ = guess_type(image_path)

        # 无法识别类型时退回通用二进制类型，保证 dataURL 合法
        if mime_type is None:
            mime_type = "application/octet-stream"

        # 以二进制模式读取图片文件并做 base64 编码，得到 UTF-8 字符串
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

        # 组装标准 dataURL 格式，供 OpenAI 视觉接口直接引用图片
        return f"data:{mime_type};base64,{base64_encoded_data}"
    
    def classify_image(self, image_path: str) -> str:
        """Analyzes the image to classify it as a medical image and determine it's type."""
        # 用途：分析图片，判定是否为医学影像并确定其类型（胸片 / 皮肤病灶 / 脑部 MRI / 其他 / 非医学）。
        # 参数: image_path - 待分类的图片路径
        # 返回值: 分类结果字典（image_type / reasoning / confidence），JSON 解析失败时返回兜底字典
        print(f"[ImageAnalyzer] Analyzing image: {image_path}")

        # 构造多模态提示词：system 设定角色 + user 携带文本指令与图片
        vision_prompt = [
            {"role": "system", "content": "You are an expert in medical imaging. Analyze the uploaded image."},
            {"role": "user", "content": [
                {"type": "text", "text": (
                    """
                    Determine if this is a medical image. If it is, classify it as:
                    'BRAIN MRI SCAN', 'CHEST X-RAY', 'SKIN LESION', or 'OTHER'. If it's not a medical image, return 'NON-MEDICAL'.
                    You must provide your answer in JSON format with the following structure:
                    {{
                    "image_type": "IMAGE TYPE",
                    "reasoning": "Your step-by-step reasoning for selecting this agent",
                    "confidence": 0.95  // Value between 0.0 and 1.0 indicating your confidence in this classification task
                    }}
                    """
                )},
                # 图片转 base64 dataURL 供视觉模型读取
                {"type": "image_url", "image_url": {"url": self.local_image_to_data_url(image_path)}}  # Correct format
            ]}
        ]
        
        # Invoke LLM to classify the image
        # 调用视觉 LLM 对图片进行分类
        response = self.vision_model.invoke(vision_prompt)

        try:
            # Ensure the response is parsed as JSON
            # 尝试将模型回复解析为 JSON 字典
            response_json = self.json_parser.parse(response.content)
            return response_json  # Returns a dictionary instead of a string
        except json.JSONDecodeError:
            # JSON 解析失败时返回兜底结果，避免上层崩溃
            print("[ImageAnalyzer] Warning: Response was not valid JSON.")
            return {"image_type": "unknown", "reasoning": "Invalid JSON response", "confidence": 0.0}

        # return response.content.strip().lower()
