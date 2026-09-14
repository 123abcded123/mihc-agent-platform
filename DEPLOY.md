# MIHC Agent 平台 — Sealos 公网部署指南

## 1. 部署拓扑

| 服务 | 端口 | 内网地址 | 公网地址 |
|---|---|---|---|
| 前端（Vue3 + nginx） | 3000 | `http://medical-agent.ns-364w8lhf:3000` | `https://fgditvzngxkq.sealoshzh.site` |
| 后端（FastAPI） | 8080 | `http://medical-agent-post.ns-364w8lhf:8080` | `https://pjivqevoppjt.sealoshzh.site` |
| MongoDB（平台数据库） | 27017 | `mongodb://root:<password>@test-db-mongodb.ns-364w8lhf.svc:27017` | 不需要公网 |

> MongoDB 外网地址可以关闭；后端在集群内用内网服务名连接。
> `deploy/sealos/backend.yaml` 含密钥，请勿公开提交（已加入 .gitignore）。

## 2. 构建镜像（GitHub Actions，无需本机 Docker）

镜像推送到 GitHub Container Registry（ghcr.io）：

1. 打开 GitHub 仓库 → **Actions** 页
2. 左侧选 **Build Images** → 右侧 **Run workflow** → 点绿色按钮执行
3. 等待两个 job（backend / frontend）变绿，镜像地址：
   - `ghcr.io/123abcded123/mihc-backend:latest`
   - `ghcr.io/123abcded123/mihc-frontend:latest`

**镜像默认私有**，Sealos 拉取前先设为公开：
GitHub 头像 → **Packages** → 点开 `mihc-backend` → **Package settings** →
右侧 Danger Zone → **Change visibility → Public**；`mihc-frontend` 同样操作。

## 3. 发布到 Sealos

1. 打开本机 `deploy/sealos/backend.yaml` 与 `deploy/sealos/frontend.yaml`
2. 把 `IMAGE_PLACEHOLDER` 替换为上面两个 ghcr.io 镜像地址
3. `backend.yaml` 修改 3 处：
   - `MIHC_AUTH_SECRET`（随机长字符串）
   - `OPENAI_API_KEY`（LLM 网关密钥）
   - `MIHC_ADMIN_PASSWORD`（管理员密码）
4. Sealos 控制台（https://hzh.sealos.run）→ **应用管理 → 新建应用** →
   粘贴 `backend.yaml` 内容发布；再建一个应用粘贴 `frontend.yaml` 发布

## 4. 验证

- 后端：https://pjivqevoppjt.sealoshzh.site/health → `healthy`
- 前端：https://fgditvzngxkq.sealoshzh.site → 注册/登录/对话/项目工作台
- 失败排查：应用详情 → Pod 日志；常见原因见 §6

## 5. 环境变量说明（后端）

| 变量 | 说明 | 云端建议值 |
|---|---|---|
| `API_PORT` | 服务端口 | `8080` |
| `MIHC_AUTH_SECRET` | HMAC token 签名密钥 | 必填强随机串 |
| `MIHC_CORS_ORIGINS` | 允许的前端来源 | `https://fgditvzngxkq.sealoshzh.site` |
| `MIHC_ALLOW_REGISTER` | 是否开放注册 | 演示 `1`，正式 `0` |
| `LLM_BASE_URL` / `LLM_MODEL` / `OPENAI_API_KEY` | LLM 网关 | 按实际填写 |
| `EMBEDDING_PROVIDER` | `local_bge_m3` / `openai_compatible` | 云端建议 `openai_compatible` |
| `EMBEDDING_API_BASE_URL` / `EMBEDDING_MODEL_NAME` / `EMBEDDING_DIM` | 远程嵌入 | 网关需支持 `/embeddings`；不支持见 §6 |
| `MILVUS_DIM` | 向量维度 | 与嵌入模型一致 |
| `RERANKER_ENABLED` | 重排模型 | 云端 `0`（RRF 得分排序） |
| `DATABASE_URL` | 关系库 | `sqlite:////data/mihc.db`（PVC） |
| `MONGO_URL` | badcase/评测 | 内网 MongoDB |
| `MILVUS_LOCAL_PATH` / `ES_LOCAL_INDEX_PATH` / `MIHC_OBJECT_STORAGE_DIR` | 本地回退存储 | 挂 `/data`（PVC 持久化） |

## 6. 常见问题

- **网关不支持 /embeddings**：改本地嵌入。给 `Dockerfile.backend` 增加
  `RUN pip install -r requirements-models.txt`，并把 `models/bge-m3` 打进镜像
  （`scripts/download_models.py` 可预下载），设置 `EMBEDDING_PROVIDER=local_bge_m3`、
  `EMBEDDING_DIM=1024`、`MILVUS_DIM=1024`。
- **Milvus 异常**：向量路自动降级，BM25 关键词路仍可回答。
- **重排未启用**：候选按 RRF 融合分排序，回答仍带引用。
- **数据丢失**：可写数据都在 `/data`（PVC）；SQLite 为演示回退，正式可换
  Sealos PostgreSQL 实例（改 `DATABASE_URL`）。
- **镜像拉取失败**：确认 §2 中 Packages 已设为 Public，或给 Sealos 配置镜像仓库密钥。
- **HTTPS**：Sealos 域名自带证书。
