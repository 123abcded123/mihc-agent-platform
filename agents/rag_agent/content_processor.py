"""
RAG 子模块组件：ContentProcessor（内容加工）。
角色：对解析后的文档做三类加工——
1) summarize_images：LLM 看图生成摘要；
2) format_document_with_images：用摘要替换 markdown 中的图片占位符；
3) chunk_document：LLM 语义切块（每块 256-512 词）。
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

class ContentProcessor:
    """
    Processes the parsed content - summarizes images, creates llm based semantic chunks

    中文说明：内容加工器。使用两个不同的 LLM：
    summarizer_model 负责图片摘要（温度 0.5），
    chunker_model 负责语义切块（温度 0.0，保证切分稳定）。
    """
    def __init__(self, config):
        """
        Initialize the response generator.

        中文说明：初始化内容加工器，从配置读取摘要模型与切块模型实例。
        
        Args:
            config: 配置对象（含 rag.summarizer_model 与 rag.chunker_model）
            llm: Large language model for image summarization
                 用于图片摘要的大语言模型
        """
        self.logger = logging.getLogger(__name__)
        self.summarizer_model = config.rag.summarizer_model     # temperature 0.5
        self.chunker_model = config.rag.chunker_model     # temperature 0.0
    
    def summarize_images(self, images: List[str]) -> List[str]:
        """
        Summarize images using the provided model, with error handling.

        中文说明：逐张调用多模态 LLM 生成图片摘要。
        单张失败不中断整体流程，以 "no image summary" 占位；
        与医学无关的图片（如按钮截图）返回 "non-informative"，后续替换时会被剔除。
        
        Args:
            images: List of image paths
                    图片路径列表（来自 doc_parser 的输出）
            
        Returns:
            List of image summaries, with placeholders for failed images
            图片摘要列表；失败图片以占位符文本填充，保证与图片数量一一对应
        """
        
        # 图片摘要提示词：要求只描述图片实际内容（图表类型、趋势等），
        # 无关图片（如界面按钮）明确返回 "non-informative"
        prompt_template = """Describe the image in detail while keeping it concise and to the point. 
                        For context, the image is part of either a medical research paper or a research paper
                        demonstrating the use of artificial intelligence techniques like
                        machine learning and deep learning in diagnosing diseases or a medical report.
                        Be specific about graphs, such as bar plots if they are present in the image.
                        Only summarize what is present in the image, without adding any extra detail or comment.
                        Summarize the image only if it is related to the context, return 'non-informative' explicitly 
                        if the image is of some button not relevant to the context."""

        # 组装多模态消息：文本提示词 + 图片 URL
        messages = [
            (
                "user",
                [
                    {"type": "text", "text": prompt_template},
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image}"},
                    },
                ],
            )
        ]

        # 构建摘要链：提示模板 → 多模态模型 → 字符串解析
        prompt = ChatPromptTemplate.from_messages(messages)
        summary_chain = prompt | self.summarizer_model | StrOutputParser()
        
        results = []
        for image in images:
            try:
                summary = summary_chain.invoke({"image": image})
                results.append(summary)
            except Exception as e:
                # Log the error if needed
                # 单张图片摘要失败时记录错误并追加占位符，
                # 保证 results 与 images 下标对齐（占位符替换依赖此对齐关系）
                print(f"Error processing image: {str(e)}")
                # Add placeholder for the failed image
                results.append("no image summary")
        
        return results
    
    def format_document_with_images(self, parsed_document: Any, image_summaries: List[str]) -> str:
        """
        Format the parsed document by replacing image placeholders with image summaries.

        中文说明：把文档导出为 markdown（图片位置用占位符标记），
        再按顺序把图片摘要替换回对应占位符位置；
        摘要为 "non-informative" 的图片直接删除占位符。
        
        Args:
            parsed_document: Parsed document from doc_parser
                             doc_parser 返回的结构化文档对象
            image_summaries: List of image summaries
                             与文档中图片顺序对应的摘要列表
            
        Returns:
            Formatted document text with image summaries
            图片已替换为文字摘要的 markdown 文本
        """
        # 约定占位符：导出时图片与分页分别标记，便于后续定位替换
        IMAGE_PLACEHOLDER = "<!-- image_placeholder -->"
        PAGE_BREAK_PLACEHOLDER = "<!-- page_break -->"
        
        # 导出为 markdown：图片处用 IMAGE_PLACEHOLDER，分页处用 PAGE_BREAK_PLACEHOLDER
        formatted_parsed_document = parsed_document.export_to_markdown(
            page_break_placeholder=PAGE_BREAK_PLACEHOLDER, 
            image_placeholder=IMAGE_PLACEHOLDER
        )
        
        # 按出现顺序逐个替换图片占位符为摘要文本
        formatted_document = self._replace_occurrences(
            formatted_parsed_document, 
            IMAGE_PLACEHOLDER, 
            image_summaries
        )
        
        return formatted_document
    
    def _replace_occurrences(self, text: str, target: str, replacements: List[str]) -> str:
        """
        Replace occurrences of a target placeholder with corresponding replacements.

        中文说明：按顺序替换占位符。
        替换内容前加 "picture_counter_N" 标记（N 为图片序号），
        供 reranker 反向提取图片引用路径；摘要为 "non-informative" 时直接移除占位符。
        占位符不足时提前终止循环而非报错。
        
        Args:
            text: Text containing placeholders
                  包含占位符的文本
            target: Placeholder to replace
                    待替换的占位符字符串
            replacements: List of replacements for each occurrence
                          与占位符依次对应的替换内容列表
            
        Returns:
            Text with replacements
            完成替换后的文本
        """
        result = text
        # 每次只替换第一个占位符（count=1），保证摘要与图片位置严格对应
        for counter, replacement in enumerate(replacements):
            if target in result:
                # 有效摘要：替换为 "picture_counter_N <摘要>"，N 用于后续图片路径还原
                if replacement.lower() != 'non-informative':
                    result = result.replace(
                        target, 
                        f'picture_counter_{counter}' + ' ' + replacement, 
                        1
                    )
                else:
                    # 无关图片：直接删除占位符，不污染检索文本
                    result = result.replace(target, '', 1)
            else:
                # Instead of raising an error, just break the loop when no more occurrences are found
                # 摘要数量多于占位符时（异常情况），直接结束循环避免越界报错
                break
        
        return result

    def chunk_document(self, formatted_document: str) -> List[str]:
        """
        Split the document into semantic chunks.

        中文说明：LLM 语义切块。流程：
        1) 先按 markdown 标题（"\\n#"）粗切分为小节并打上 <|start_chunk_X|> 标记；
        2) 调用 LLM 判断相邻小节是否同主题，输出应切分的位置；
        3) 按 LLM 建议把同主题小节合并成 256-512 词的最终块。
        
        Args:
            formatted_document: Formatted document text
                                已替换图片摘要的 markdown 文档
            model: AzureChatOpenAI model instance (will create one if not provided)
                   用于切块的 AzureChatOpenAI 模型实例（未提供时内部创建）
            
        Returns:
            List of document chunks
            语义块列表（每块 256-512 词）
        """
        
        # Split by section boundaries
        # 按标题粗切分：标题前换行处断开，得到带层级结构的粗块
        SPLIT_PATTERN = "\n#"
        chunks = formatted_document.split(SPLIT_PATTERN)
        
        # 为每个粗块打上起止标记，供 LLM 引用块编号给出切分建议
        chunked_text = ""
        for i, chunk in enumerate(chunks):
            if chunk.startswith("#"):
                chunk = f"#{chunk}"  # add the # back to the chunk
            chunked_text += f"<|start_chunk_{i}|>\n{chunk}\n<|end_chunk_{i}|>\n"
        
        # LLM-based semantic chunking
        # 语义切块提示词：要求 LLM 识别主题边界、保持 256-512 词/块，
        # 输出格式固定为 "split_after: 3, 5" 便于程序解析
        CHUNKING_PROMPT = """
        You are an assistant specialized in splitting text into semantically consistent sections. 
        
        Following is the document text:
        <document>
        {document_text}
        </document>
        
        <instructions>
        Instructions:
            1. The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number.
            2. Identify points where splits should occur, such that consecutive chunks of similar themes stay together.
            3. Each chunk must be between 256 and 512 words.
            4. If chunks 1 and 2 belong together but chunk 3 starts a new topic, suggest a split after chunk 2.
            5. The chunks must be listed in ascending order.
            6. Provide your response in the form: 'split_after: 3, 5'.
        </instructions>
        
        Respond only with the IDs of the chunks where you believe a split should occur.
        YOU MUST RESPOND WITH AT LEAST ONE SPLIT.
        """.strip()
        
        # 填充提示词并调用切块模型（温度 0.0，输出稳定）
        formatted_chunking_prompt = CHUNKING_PROMPT.format(document_text=chunked_text)
        chunking_response = self.chunker_model.invoke(formatted_chunking_prompt).content
        
        return self._split_text_by_llm_suggestions(chunked_text, chunking_response)
    
    def _split_text_by_llm_suggestions(self, chunked_text: str, llm_response: str) -> List[str]:
        """
        Split text according to LLM suggested split points.

        中文说明：按 LLM 返回的切分点把粗块合并为最终语义块。
        LLM 未给出有效建议时，整篇文档作为单块返回。
        
        Args:
            chunked_text: Text with chunk markers
                          带 <|start_chunk_X|>/<|end_chunk_X|> 标记的文本
            llm_response: LLM response with split suggestions
                          LLM 返回的 "split_after: ..." 切分建议
            
        Returns:
            List of document chunks
            合并后的语义块列表
        """
        # Extract split points from LLM response
        # 从 LLM 回复中解析切分点编号列表（如 "3, 5" → [3, 5]）
        split_after = [] 
        if "split_after:" in llm_response:
            split_points = llm_response.split("split_after:")[1].strip()
            split_after = [int(x.strip()) for x in split_points.replace(',', ' ').split()] 

        # If no splits were suggested, return the whole text as one section
        # 未解析出有效切分点时，整篇作为单块返回（容错降级）
        if not split_after:
            return [chunked_text]

        # Find all chunk markers in the text
        # 用正则提取全部粗块及其编号，捕获组 \1 保证起止标记编号一致
        chunk_pattern = r"<\|start_chunk_(\d+)\|>(.*?)<\|end_chunk_\1\|>"
        chunks = re.findall(chunk_pattern, chunked_text, re.DOTALL)

        # Group chunks according to split points
        # 按切分点分组：遇到编号在 split_after 中的块时，结束当前 section
        sections = []
        current_section = [] 

        for chunk_id, chunk_text in chunks:
            current_section.append(chunk_text)
            if int(chunk_id) in split_after:
                sections.append("".join(current_section).strip())
                current_section = [] 
        
        # Add the last section if it's not empty
        # 收尾：最后一个切分点之后的剩余内容也要并入结果
        if current_section:
            sections.append("".join(current_section).strip())

        return sections
