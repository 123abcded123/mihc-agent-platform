# Dockerfile —— 多智能体医疗助手的容器镜像构建文件
#
# 本文件在系统中的角色：
# 1. 定义应用的标准运行环境（Python 3.11 + 系统级依赖），保证任何机器上行为一致；
# 2. 构建分层利用 Docker 缓存：先拷贝 requirements.txt 安装依赖，再拷贝源码，
#    源码变动时无需重新下载依赖，显著加快二次构建；
# 3. 预装图像处理（OpenCV）、音频转码（ffmpeg）、XML 解析（lxml）等底层库，
#    支撑医学图像分析与语音功能；
# 4. 通过 HEALTHCHECK 对接应用的 /health 接口，供编排系统自动重启不健康容器；
# 5. 定义容器启动命令：python app.py（即启动 FastAPI 服务）。

# Base image with Python 3.11
# 基础镜像：精简版 Python 3.11（slim 体积小，减少镜像尺寸与攻击面）
FROM python:3.11-slim

# Set working directory
# 设定工作目录：后续 COPY/RUN/CMD 都相对 /app 执行
WORKDIR /app

# Install system dependencies
# 安装系统级依赖（一次性执行并清理 apt 缓存，控制镜像体积）：
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \            # 音频转码工具：pydub 转 webm→mp3 依赖它
    build-essential \   # 编译工具链：部分 Python 包需从源码编译
    # OpenCV dependencies
    # OpenCV 运行所需共享库：缺了会导致 cv2 导入即崩溃
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    # Image processing dependencies
    # 图像编解码库：支撑 PNG/JPEG 图片读写
    libpng-dev \
    libjpeg-dev \
    # For lxml
    # lxml（解析 XML/HTML 文档）编译依赖
    libxml2-dev \
    libxslt1-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
# 先只拷贝依赖清单：利用 Docker 分层缓存，依赖未变时跳过重新安装
COPY requirements.txt .

# Install Python dependencies
# 安装 Python 依赖：--no-cache-dir 避免把 pip 缓存打进镜像
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
# 拷贝应用源码到容器工作目录
COPY . .

# Create necessary directories
# 预创建运行时目录（上传/分割结果/语音/数据），避免应用启动时因缺目录报错
RUN mkdir -p uploads/backend uploads/frontend uploads/skin_lesion_output uploads/speech data

# Expose port
# 声明服务端口：告知编排工具容器监听 8000
EXPOSE 8000

# Set environment variable for Python to run in unbuffered mode
# 关闭 Python 输出缓冲：日志/print 实时可见，便于容器排障
ENV PYTHONUNBUFFERED=1

# Set healthcheck
# 健康检查：每 30s 探测一次 /health，连续失败 3 次则标记不健康
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
# 容器启动命令：运行 FastAPI 应用（app.py 内部读取配置的 host/port）
CMD ["python", "app.py"]
