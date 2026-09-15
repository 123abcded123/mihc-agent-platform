"""
命令行 mIHC 文献下载入库工具

用法：
  python scripts/download_mihc_literature.py --query "mIHC tumor microenvironment" --max 10 --download 5
  python scripts/download_mihc_literature.py --query "multiplex immunofluorescence CD8" --max 10 --download 3 --no-ingest
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="mIHC 文献自动下载入库")
    parser.add_argument("--query", default="mIHC tumor immune microenvironment", help="PubMed 检索词")
    parser.add_argument("--max", dest="max_results", type=int, default=10, help="检索结果数上限")
    parser.add_argument("--download", dest="max_download", type=int, default=5, help="下载开放获取文献数上限")
    parser.add_argument("--no-ingest", action="store_true", help="只下载不入库")
    args = parser.parse_args()

    if args.no_ingest:
        from mihc.literature import PubMedLiterature
        from config import Config
        downloader = PubMedLiterature(Config().mihc.literature_dir)
        files = downloader.download(args.query, args.max_results, args.max_download)
        for f in files:
            logger.info("已下载: %s", f["file_path"])
        logger.info("共下载 %d 篇", len(files))
        return

    from config import Config
    from services.service import MIHCPlatform

    platform = MIHCPlatform(Config())
    result = platform.ingest_literature(query=args.query, max_results=args.max_results,
                                        max_download=args.max_download)
    logger.info("下载 %(downloaded)s 篇，入库 %(ingested)s 篇", result)
    for d in result["details"]:
        logger.info("  %s -> %s", d["file"], d.get("status", "?"))


if __name__ == "__main__":
    main()
