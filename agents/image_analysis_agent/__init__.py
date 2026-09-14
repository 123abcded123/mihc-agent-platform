"""
医学影像分析 Agent 门面模块。

职责：
- 作为"医学影像分析"模块的统一入口（门面 Facade），对外屏蔽内部各子 Agent 的实现细节；
- 组合三个子 Agent：图片类型分类器（GPT-4o 视觉）、新冠胸片二分类器（DenseNet121）、皮肤病灶分割器（U-Net）；
- 为上游编排层提供统一的分析接口，按图片类型路由到对应的专业子 Agent。
"""
from .image_classifier import ImageClassifier
from .chest_xray_agent.covid_chest_xray_inference import ChestXRayClassification
# 脑肿瘤 Agent 尚未实现，其导入语句被注释（参见 brain_tumor_agent/brain_tumor_inference.py 中的 # TBD）
# from .brain_tumor_agent.brain_tumor_inference import BrainTumorAgent
from .skin_lesion_agent.skin_lesion_inference import SkinLesionSegmentation

class ImageAnalysisAgent:
    """
    Agent responsible for processing image uploads and classifying them as medical or non-medical, and determining their type.
    """
    # 医学影像分析门面类：
    # 用途：接收图片上传请求，先判定图片是否为医学影像及其类型，再分发到对应的专业子 Agent 执行。
    
    def __init__(self, config):
        """
        初始化影像分析门面。

        参数:
            config: 全局配置对象，其 medical_cv 子节提供视觉 LLM、各子模型权重路径及分割结果输出路径。
        """
        # 初始化图片类型分类器：使用配置中指定的视觉 LLM（GPT-4o Vision）
        self.image_classifier = ImageClassifier(vision_model=config.medical_cv.llm)
        # 初始化新冠胸片二分类 Agent：加载 DenseNet121 权重
        self.chest_xray_agent = ChestXRayClassification(model_path=config.medical_cv.chest_xray_model_path)
        # 脑肿瘤 Agent 尚未实现，暂不实例化
        # self.brain_tumor_agent = BrainTumorAgent()
        # 初始化皮肤病灶分割 Agent：加载 U-Net 权重
        self.skin_lesion_agent = SkinLesionSegmentation(model_path=config.medical_cv.skin_lesion_model_path)
        # 皮肤病灶分割结果（掩码叠加图）的保存路径
        self.skin_lesion_segmentation_output_path = config.medical_cv.skin_lesion_segmentation_output_path
    
    # classify image
    # 图片分类：判定是否为医学影像及其类型（胸片 / 皮肤病灶 / 脑部 MRI / 其他 / 非医学）
    def analyze_image(self, image_path: str) -> str:
        """Classifies images as medical or non-medical and determines their type."""
        # 用途：分析图片并返回分类结果 JSON（image_type / reasoning / confidence）。
        # 参数: image_path - 待分析的本地图片路径
        # 返回值: 图片类型分类结果字典
        return self.image_classifier.classify_image(image_path)
    
    # chest x-ray agent
    # 胸片分类：调用新冠胸片 Agent 进行 covid19 / normal 二分类
    def classify_chest_xray(self, image_path: str) -> str:
        # 用途：对胸部 X 光片做新冠二分类。
        # 参数: image_path - 胸片图片路径
        # 返回值: 预测类别字符串（covid19 / normal），异常时返回 None
        return self.chest_xray_agent.predict(image_path)
    
    # # brain tumor agent
    # 脑肿瘤分类：Agent 尚未实现，接口预留但暂不可用
    # def classify_brain_tumor(self, image_path: str) -> str:
    #     return self.brain_tumor_agent.predict(image_path)
    
    # skin lesion agent
    # 皮肤病灶分割：调用皮肤病灶 Agent 生成分割掩码并叠加到原图保存
    def segment_skin_lesion(self, image_path: str) -> str:
        # 用途：对皮肤病灶图片执行 U-Net 分割，并将掩码叠加图保存到配置的输出路径。
        # 参数: image_path - 皮肤病灶图片路径
        # 返回值: 叠加图生成结果（True / 异常抛出）
        return self.skin_lesion_agent.predict(image_path, self.skin_lesion_segmentation_output_path)
