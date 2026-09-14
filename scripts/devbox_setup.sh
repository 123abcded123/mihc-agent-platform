#!/usr/bin/env bash
# MIHC 后端 — Sealos DevBox 一键部署脚本
# 用法：bash scripts/devbox_setup.sh
# 完成后服务运行在 0.0.0.0:8080（DevBox 已映射公网域名）
set -e

REPO="https://github.com/123abcded123/mihc-agent-platform"
APP_DIR="$HOME/mihc-platform"

echo "==> [1/4] 检查 Python 环境"
if ! command -v python3 >/dev/null 2>&1; then
    echo "未找到 python3，请在 DevBox 模板中切换为 Python 环境后重试"
    exit 1
fi
python3 --version
command -v pip3 >/dev/null || python3 -m ensurepip --upgrade

echo "==> [2/4] 拉取代码"
if [ -d "$APP_DIR/.git" ]; then
    (cd "$APP_DIR" && git pull --rebase origin main)
else
    git clone "$REPO" "$APP_DIR"
fi

echo "==> [3/4] 安装依赖（约 2-3 分钟）"
cd "$APP_DIR"
python3 -m pip install --upgrade pip -q
pip3 install -r requirements.txt -q

echo "==> [4/4] 配置与启动"
if [ ! -f .env ]; then
    cp .env.example .env
    echo "已生成 .env，请按下方提示修改后再运行：bash scripts/devbox_run.sh"
    exit 0
fi
bash scripts/devbox_run.sh
