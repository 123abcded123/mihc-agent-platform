"""
RAG 子模块组件：MedicalDocParser（PDF 文档解析）。
角色：使用 Docling 将医疗研究 PDF 解析为结构化文档对象，
同时导出页面图片、表格图片与插图图片到本地目录，
供后续图片摘要与向量入库管线使用。
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Docling 相关导入：输入格式、PDF 管线选项（含表格模式/OCR/图片描述）
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    TableFormerMode, 
    RapidOcrOptions, 
    smolvlm_picture_description
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem

class MedicalDocParser:
    """
    Handles parsing of medical research documents using docling.

    中文说明：医疗文档解析器。封装 Docling 转换器，
    输出结构化文档（含标题层级、表格、公式、图片引用）与图片文件列表。
    """
    def __init__(self):
        # 解析器无外部依赖，仅初始化日志；真正的模型加载发生在 parse_document 中
        self.logger = logging.getLogger(__name__)
        self.logger.info("Medical Document Parser initialized!")

    def parse_document(
            self,
            document_path: str,
            output_dir: str,
            image_resolution_scale: float = 2.0,
            do_ocr: bool = True,
            do_tables: bool = True,
            do_formulas: bool = True,
            do_picture_desc: bool = False
        ) -> Tuple[Any, List[str]]:
        """
        Parse the document and extract structured content and images.

        中文说明：解析 PDF 文档：
        1) 配置 Docling 管线（页图/插图/OCR/表格/公式/图片描述）；
        2) 转换文档并导出页面、表格、插图 PNG 到 output_dir；
        3) 返回结构化文档对象 + 插图图片路径列表。
        
        Args:
            document_path: Path to the document to parse
                            待解析的 PDF 文档路径
            output_dir: Directory to save extracted images
                            导出图片（页面/表格/插图）的保存目录
            image_resolution_scale: Resolution scale for extracted images
                            导出图片的分辨率放大倍数（默认 2 倍）
            do_ocr: Enable OCR processing
                            是否启用 OCR 文本识别（扫描版 PDF 需要）
            do_tables: Enable table structure extraction
                            是否启用表格结构抽取
            do_formulas: Enable formula enrichment
                            是否启用公式富化（公式转 LaTeX 等）
            do_picture_desc: Enable picture description generation
                            是否启用图片描述生成（本流程未使用，摘要由外部 LLM 完成）
            
        Returns:
            Tuple containing (parsed_document, list_of_image_paths)
            元组：(结构化文档对象, 插图图片路径列表)
        """
        # Create output directory if it doesn't exist
        # 确保输出目录存在，避免后续写文件失败
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Configure pipeline options
        # 配置 PDF 解析管线：开启页面/插图图片导出、OCR、表格结构与公式富化
        pipeline_options = PdfPipelineOptions(
            generate_page_images=True,
            generate_picture_images=True,
            images_scale=image_resolution_scale,
            do_ocr=do_ocr,
            do_table_structure=do_tables,
            do_formula_enrichment=do_formulas,
            do_picture_description=do_picture_desc
        )
        
        # Set table structure mode
        # 表格结构识别采用高精度模式（牺牲速度换取表格还原质量）
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE    # Can choose between FAST and ACCURATE
        
        # Initialize document converter
        # 初始化转换器，指定 PDF 输入使用上述管线选项
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        
        # Convert document
        # 执行文档转换（CPU/GPU 密集，耗时较长）
        conversion_res = converter.convert(document_path)
        
        # Get document filename
        # 提取文件名主干（不含扩展名），用于图片文件命名
        doc_filename = conversion_res.input.file.stem
        
        # Save page images
        # 逐页导出整页渲染图，文件名形如 <文档名>-<页码>.png
        for page_no, page in conversion_res.document.pages.items():
            page_image_filename = output_dir_path / f"{doc_filename}-{page_no}.png"
            with page_image_filename.open("wb") as fp:
                page.image.pil_image.save(fp, format="PNG")
        
        # Save images of figures and tables
        # 导出表格与插图图片；计数器用于生成唯一文件名
        table_counter = 0
        picture_counter = 0
        image_paths = []
        
        # 遍历文档元素树：表格元素与插图元素分别导出 PNG
        for element, _level in conversion_res.document.iterate_items():
            if isinstance(element, TableItem):
                table_counter += 1
                element_image_filename = output_dir_path / f"{doc_filename}-table-{table_counter}.png"
                with element_image_filename.open("wb") as fp:
                    element.get_image(conversion_res.document).save(fp, "PNG")
                    
            if isinstance(element, PictureItem):
                # 插图文件名固定为 picture-<序号>，序号在文档中的出现顺序
                # 该序号会被 content_processor 写入 picture_counter_N 标记，供重排器反查图片路径
                picture_path = f"{doc_filename}-picture-{picture_counter}.png"
                element_image_filename = output_dir_path / picture_path
                with element_image_filename.open("wb") as fp:
                    element.get_image(conversion_res.document).save(fp, "PNG")
                
                # Add path to the list of images
                # 记录插图路径，供后续 LLM 摘要使用
                image_paths.append(str(element_image_filename))
                picture_counter += 1
        
        # Extract images for summarization
        # 再次从文档对象中提取图片引用 URI，作为摘要输入（与上面导出的路径配套）
        images = []
        for picture in conversion_res.document.pictures:
            ref = picture.get_ref().cref
            image = picture.image
            if image:
                images.append(str(image.uri))
        
        return conversion_res.document, images
