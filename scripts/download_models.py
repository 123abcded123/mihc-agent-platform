"""
模型预下载脚本（BGE-M3 嵌入 + BGE-Reranker 重排）

从 HF 镜像（HF_ENDPOINT 可配，默认 hf-mirror.com）下载模型到本地 ./models/，
供平台本地加载（不依赖 huggingface_hub 运行时下载，支持断点续传）。

用法：
  python scripts/download_models.py                  # 全部模型
  python scripts/download_models.py --only bge-m3    # 只下载嵌入模型
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import requests

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

MODEL_FILES = {
    "bge-m3": [
        "config.json",
        "config_sentence_transformers.json",
        "sentence_bert_config.json",
        "modules.json",
        "pytorch_model.bin",
        "sentencepiece.bpe.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "1_Pooling/config.json",
    ],
    "bge-reranker-base": [
        "config.json",
        "model.safetensors",
        "sentencepiece.bpe.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ],
}


def download_file(url: str, dest: str, expected_size: int | None = None, resume: bool = True) -> bool:
    """下载单个文件（断点续传 + 进度显示 + 绕过系统代理）。"""
    if os.path.exists(dest) and expected_size and os.path.getsize(dest) >= expected_size:
        print(f"  [跳过] {dest} 已存在")
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    headers = {}
    offset = 0
    if resume and os.path.exists(dest):
        offset = os.path.getsize(dest)
        headers["Range"] = f"bytes={offset}-"

    # 绕过系统代理（本机代理对 hf-mirror 不稳定，直连可用）
    session = requests.Session()
    session.trust_env = False
    last_error = None
    for attempt in range(5):
        try:
            with session.get(url, stream=True, timeout=120, headers=headers) as r:
                if r.status_code == 416:
                    # Range 越界：文件其实已完整，或需要重新下载
                    os.remove(dest)
                    headers.pop("Range", None)
                    offset = 0
                    continue
                if r.status_code not in (200, 206):
                    print(f"  [失败] HTTP {r.status_code}: {url}")
                    return False
                mode = "ab" if r.status_code == 206 else "wb"
                total = int(r.headers.get("Content-Length", 0)) + (offset if mode == "ab" else 0)
                done = offset
                start = time.time()
                with open(dest, mode) as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = done / total * 100
                            speed = done / max(1e-6, time.time() - start) / 1e6
                            print(f"\r  {os.path.basename(dest)}: {pct:.1f}% ({done / 1e6:.0f}/{total / 1e6:.0f} MB, {speed:.1f} MB/s)", end="")
                print()
                return True
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"\n  [重试 {attempt + 1}/5] 下载中断: {exc}")
            time.sleep(3)
    print(f"  [失败] {dest}: {last_error}")
    return False


def main():
    parser = argparse.ArgumentParser(description="预下载 BGE-M3 / BGE-Reranker 模型")
    parser.add_argument("--only", choices=list(MODEL_FILES), default=None)
    parser.add_argument("--dest", default="./models")
    args = parser.parse_args()

    models = [args.only] if args.only else list(MODEL_FILES)
    for model in models:
        print(f"下载模型: {model}")
        for rel in MODEL_FILES[model]:
            url = f"{HF_ENDPOINT}/{ {'bge-m3': 'BAAI/bge-m3', 'bge-reranker-base': 'BAAI/bge-reranker-base'}[model] }/resolve/main/{rel}"
            dest = os.path.join(args.dest, model, rel)
            if not download_file(url, dest):
                print(f"下载失败: {rel}，请检查网络后重试（支持断点续传）")
                sys.exit(1)
        print(f"{model} 完成 -> {os.path.join(args.dest, model)}")
    print("全部模型就绪。")


if __name__ == "__main__":
    main()
