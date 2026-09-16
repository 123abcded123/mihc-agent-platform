"""
清空 RAG 知识库：PostgreSQL/SQLite 文档元数据 + BM25 关键词索引 + Milvus 集合。
客户项目数据（projects/samples/markers/files/analysis_runs）不受影响。
用法：python scripts/clear_knowledge_base.py [--yes]
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="清空 RAG 知识库")
    parser.add_argument("--yes", action="store_true", help="跳过确认直接执行")
    args = parser.parse_args()

    if not args.yes:
        answer = input("将删除全部知识库文档/片段/向量，仅保留项目数据。确认？(y/N): ").strip().lower()
        if answer != "y":
            print("已取消")
            return

    from config import Config
    from infra.postgres import Database
    from rag.retriever.es_store import KeywordStore
    from rag.retriever.milvus_store import MilvusStore

    config = Config()
    db = Database(config.postgres.url)
    db.clear_knowledge()
    logger.info("PG/SQLite documents+chunks 已清空")

    keyword = KeywordStore(config)
    keyword.clear()
    logger.info("关键词索引已清空")

    milvus = MilvusStore(config)
    milvus.clear()
    logger.info("Milvus 集合已删除")

    print("知识库已清空（客户项目数据未受影响）")


if __name__ == "__main__":
    main()
