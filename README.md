# MIHC Agent · 客户多组学 mIHC 科研服务 Agent 平台

面向医疗科研客户（研究肿瘤免疫微环境的科研团队）的受控智能 Agent 平台：
**产品咨询、实验设计、表格/多组学分析、文献推荐、报告生成**，支持项目、样本、
marker、客户文件与确定性统计工具，答案带引用、风险提示与审计链路。

> 原则：模型负责“判断该调用什么工具、解释已验证结果”，不替代工具计算
> `p_value`、阳性率或样本对应关系；权限与安全检查不依赖 Prompt。

## 核心能力

| 分支 | 说明 |
|---|---|
| 产品咨询 | HyperView/Panel/抗体库/样本接收等公开资料问答（带来源） |
| 实验设计 | 检索 SOP/Panel，生成方案草稿 + 对照/重复数/终点指标校验 |
| 表格分析 | CSV/Excel 校验（sample_id 对齐、阈值、分组）→ 确定性统计（阳性率/组间比较/组学关联） |
| 图像分析 | 图像分支预留：需通道映射与模型版本，否则转人工 |
| 文献推荐 | Query Rewrite/HyDE → ES/BM25 + Milvus/BGE-M3 双路召回 → RRF + BGE-Reranker |
| 报告生成 | 依赖已通过质控的 `analysis_run`，区分观察/统计/假设/局限 |

## 技术栈

- **后端**：FastAPI + LangGraph 多 Agent 编排 + Pydantic
- **检索**：ES/BM25 + Milvus(milvus-lite 回退) + RRF 融合 + BGE-Reranker（可降级）
- **意图**：规则分类器（可替换为 BERT 微调模型）+ 置信度路由 + 澄清/人工分支
- **存储**：PostgreSQL/SQLite（项目/样本/marker/文件/分析运行/审计），
  Redis/fakeredis（会话），MongoDB/JSON（badcase/评测），本地对象存储（客户文件）
- **安全**：PBKDF2 登录 + HMAC token、DFA 敏感词、Prompt 护栏、租户/项目权限
- **观测**：OpenTelemetry / Prometheus / Langfuse（可选）

## 快速开始（本地）

```bash
pip install -r requirements.txt
# 可选：本地 BGE 嵌入/重排（无远程嵌入端点时）
pip install -r requirements-models.txt
python scripts/download_models.py   # 下载 bge-m3 / bge-reranker-base 到 models/

cp .env.example .env                # 按需修改 LLM/嵌入/存储配置
python app.py                       # 默认 http://127.0.0.1:8000

# 前端（开发模式，代理到 8000）
cd frontend && npm install && npm run dev
```

验证：`GET /health` 返回 healthy；`POST /api/v1/chat` 对话；
注册登录后使用项目工作台（`POST /api/v1/projects` 等接口）。

## 目录结构

```
app.py               # FastAPI 入口（对话/鉴权/项目工作台/知识库）
config.py            # 集中配置（多后端回退）
core/                # LLM 工厂、嵌入、数据模型、异常
services/            # 平台服务编排 + 登录鉴权
mihc/                # mIHC 领域：意图分类、路由、对象存储、确定性表格分析
agents/              # LangGraph 多 Agent（文献/知识/检验/数据/实验/合并/引用检查）
rag/                 # 入库管线 + 混合检索（ES/BM25 + Milvus + RRF + Reranker）
security/            # DFA 敏感词 + Prompt 护栏
infra/               # PostgreSQL/SQLite、Redis/fakeredis、Mongo/JSON
observability/       # OTel / Langfuse / Prometheus
eval/                # Harness 评测（HitRate/Recall/MRR/忠实性/引用/安全）
finetune/            # BERT 意图微调、SFT/LoRA、DPO 脚本
frontend/            # Vue3 + Element Plus
deploy/              # Dockerfile、nginx、Sealos/K8s 部署清单
```

## 部署（Sealos）

公网部署指南见 [DEPLOY.md](DEPLOY.md)：GitHub Actions 构建镜像 → Sealos 发布。
