
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from services.service import MIHCPlatform

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED = (".pdf", ".docx", ".md", ".txt")

def main():
    parser = argparse.ArgumentParser(description="MIHC 知识库文档入库")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="单个文档路径")
    group.add_argument("--dir", help="目录（递归处理其中的支持文件）")
    parser.add_argument("--version", default="", help="文档版本，如 2026-01")
    parser.add_argument("--source", default="", help="来源（DOI/原始路径），默认取文件路径")
    parser.add_argument("--permission", default="research_team", help="权限标签")
    args = parser.parse_args()

    config = Config()
    platform = MIHCPlatform(config)

    files = []
    if args.file:
        files = [args.file]
    else:
        for root, _dirs, names in os.walk(args.dir):
            for name in names:
                if name.lower().endswith(SUPPORTED):
                    files.append(os.path.join(root, name))

    total_chunks = 0
    for f in files:
        logger.info("入库: %s", f)
        result = platform.ingest_document(
            f, version=args.version, source=args.source or f, permission=args.permission)
        total_chunks += result.chunks
        logger.info("  -> doc_id=%s chunks=%d embedding=%s", result.doc_id, result.chunks, "BGE-M3")

    logger.info("完成：%d 个文件，共 %d 个片段（Milvus + 关键词库 + PostgreSQL 元数据）",
                len(files), total_chunks)

if __name__ == "__main__":
    main()
