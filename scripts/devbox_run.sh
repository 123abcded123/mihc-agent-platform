#!/usr/bin/env bash
# MIHC 后端 — DevBox 启动/重启脚本
set -e
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$APP_DIR"
mkdir -p data

# 杀掉旧进程后后台启动，日志写 data/server.log
pkill -f "uvicorn app:app" 2>/dev/null || true
sleep 1

nohup python3 -m uvicorn app:app --host 0.0.0.0 --port "${API_PORT:-8080}" \
    > data/server.log 2>&1 &

echo "已启动：端口 ${API_PORT:-8080}，日志 $APP_DIR/data/server.log"
echo "查看日志：tail -f $APP_DIR/data/server.log"
