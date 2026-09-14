"""
ingest_rag_data.py —— RAG 知识库数据入库脚本（命令行工具）

本文件在系统中的角色：
1. 离线数据管道：把医疗文档（PDF、文本等）解析、切分、向量化后写入 Qdrant 向量库，
   为 RAG_AGENT 提供可检索的知识底座；属于独立的运维/初始化流程，不参与在线请求处理；
2. 命令行入口：支持 --file（单个文件）或 --dir（整个目录）两种入库方式；
3. 结果反馈：以 JSON 形式打印入库结果，并以返回码（成功/失败）供自动化脚本判断。

使用示例：
    python ingest_rag_data.py --file ./data/docs/guide.pdf
    python ingest_rag_data.py --dir ./data/docs_db
"""

import sys
import json
import logging
from pathlib import Path

import warnings
# 屏蔽第三方库的无害警告：避免解析 PDF 等文档时输出大量干扰日志
warnings.filterwarnings('ignore')

# Set up logging
# 配置日志：统一输出格式，便于追踪入库过程中各阶段的状态
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add project root to path if needed
# 将项目根目录加入 sys.path：保证无论从哪个目录运行本脚本，都能导入项目内的 agents 与 config 模块
sys.path.append(str(Path(__file__).parent.parent))

# Import your components
# 导入 RAG 核心组件与全局配置
from agents.rag_agent import MedicalRAG
from config import Config

import argparse

# Initialize parser
# 初始化命令行参数解析器
parser = argparse.ArgumentParser(description="Process some command-line arguments.")

# Add arguments
# --file：入库单个文件；--dir：入库整个目录（两者至少提供一个）
parser.add_argument("--file", type=str, required=False, help="Enter file path to ingest")
parser.add_argument("--dir", type=str, required=False, help="Enter directory path of files to ingest")

# Parse arguments
# 解析命令行参数
args = parser.parse_args()

# Load configuration
# 加载全局配置（含嵌入模型、向量库路径等），RAG 实例化需要这些参数
config = Config()

# 实例化 MedicalRAG：负责文档解析、切分、向量化与写入 Qdrant 的全部工作
rag = MedicalRAG(config)

# document ingestion
def data_ingestion():
    """执行文档入库的主函数。

    输入：命令行参数 args.file（单文件路径）或 args.dir（目录路径）
    输出：bool —— 入库是否成功（True/False），供主程序判断并打印结果
    流程：根据参数选择调用 rag.ingest_file 或 rag.ingest_directory，并把结果以 JSON 打印
    """

    if args.file:
        # Define path to file
        file_path = args.file
        # Process and ingest the file
        # 入库单个文件：解析 → 切块 → 向量化 → 写入 Qdrant
        result = rag.ingest_file(file_path)
    elif args.dir:
        # Define path to dir
        dir_path = args.dir
        # Process and ingest the files
        # 入库整个目录：遍历目录内所有文档逐一入库
        result = rag.ingest_directory(dir_path)

    # 以可读的 JSON 格式打印入库结果（含成功/失败、文档数、错误信息等）
    print("Ingestion result:", json.dumps(result, indent=2))

    # 返回成功标志：供 __main__ 中据此提示用户
    return result["success"]

# Run tests
if __name__ == "__main__":
   
    print("\nIngesting document(s)...")

    # 执行入库并获取结果
    ingestion_success = data_ingestion()
    
    if ingestion_success:
        # 仅成功时输出提示；失败时依靠上方 JSON 日志定位原因
        print("\nSuccessfully ingested the documents.")
