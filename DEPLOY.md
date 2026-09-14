# MIHC Agent 平台 — Sealos 公网部署指南

## 1. 部署拓扑

| 服务 | 端口 | 内网地址 | 公网地址 |
|---|---|---|---|
| 前端（Vue3 + nginx） | 3000 | `http://medical-agent.ns-364w8lhf:3000` | `https://fgditvzngxkq.sealoshzh.site` |
| 后端（FastAPI） | 8080 | `http://medical-agent-post.ns-364w8lhf:8080` | `https://pjivqevoppjt.sealoshzh.site` |
| MongoDB（平台数据库） | 27017 | `mongodb://root:<password>@test-db-mongodb.ns-364w8lhf.svc:27017` | **无需开启公网**（省钱） |

> MongoDB 外网地址不需要，后端在集群内用内网服务名连接；账号密码已写入
> `deploy/sealos/backend.yaml` 的 `MONGO_URL`（该文件含密钥，请勿公开提交到公开仓库）。

## 2. 构建镜像（本地执行）

```bash
# 后端：FastAPI 镜像（监听 8080）
docker build -f Dockerfile.backend -t <registry>/mihc-backend:latest .
docker push <registry>/mihc-backend:latest

# 前端：Vue3 + nginx 镜像（监听 3000）
# 关键：构建时把后端公网地址写进前端产物
docker build --build-arg VITE_API_BASE=https://pjivqevoppjt.sealoshzh.site \
  -f Dockerfile.frontend -t <registry>/mihc-frontend:latest .
docker push <registry>/mihc-frontend:latest
```

`<registry>` 可为 Docker Hub（`docker.io/<你的账号>`）或 Sealos 自带的私有镜像仓库。

## 3. 部署到 Sealos

1. 修改 `deploy/sealos/backend.yaml` 与 `deploy/sealos/frontend.yaml`：
   - 替换 `IMAGE_PLACEHOLDER` 为第 2 步推送到仓库的镜像地址；
   - 修改 `MIHC_AUTH_SECRET`（生产密钥，随机长字符串）；
   - 可选 `MIHC_ADMIN_USER` / `MIHC_ADMIN_PASSWORD`（首次启动自动建管理员）；
   - 填写 `OPENAI_API_KEY`（LLM 网关密钥）。
2. 在 Sealos 控制台「应用」中导入两个 yaml，或本地用 kubectl 应用。
3. 等待 Pod 就绪后访问：
   - 前端：https://fgditvzngxkq.sealoshzh.site
   - 后端探活：https://pjivqevoppjt.sealoshzh.site/health

## 4. 环境变量说明（后端）

| 变量 | 说明 | 云端建议值 |
|---|---|---|
| `API_PORT` | 服务端口 | `8080` |
| `MIHC_AUTH_SECRET` | HMAC token 签名密钥 | 必填强随机串 |
| `MIHC_CORS_ORIGINS` | 允许的前端来源 | `https://fgditvzngxkq.sealoshzh.site` |
| `MIHC_ALLOW_REGISTER` | 是否开放注册 | 演示 `1`，正式可 `0` |
| `LLM_BASE_URL` / `LLM_MODEL` / `OPENAI_API_KEY` | LLM 网关 | 按实际网关填写 |
| `EMBEDDING_PROVIDER` | `local_bge_m3` / `openai_compatible` | 云端建议 `openai_compatible` |
| `EMBEDDING_API_BASE_URL` / `EMBEDDING_MODEL_NAME` / `EMBEDDING_DIM` | 远程嵌入端点 | 需网关支持 `/embeddings`；不支持时改用本地模式（见 §6） |
| `MILVUS_DIM` | 向量维度 | 必须与嵌入模型维度一致 |
| `RERANKER_ENABLED` | 重排模型 | 云端 `0`（降级为 RRF 得分排序） |
| `DATABASE_URL` | 关系库 | `sqlite:////data/mihc.db`（PVC） |
| `MONGO_URL` | badcase/评测 | 内网 MongoDB 地址 |
| `MILVUS_LOCAL_PATH` / `ES_LOCAL_INDEX_PATH` / `MIHC_OBJECT_STORAGE_DIR` | 本地回退存储 | 均挂到 `/data`（PVC 持久化） |

## 5. 鉴权行为

- `/api/v1/chat`：**公开**（产品咨询、通用问答）；携带 Bearer token 时绑定登录用户。
- 其余接口（项目/文件/分析/知识库/会话/badcase/评测）：**必须登录**，未登录返回 401。
- 密码使用 PBKDF2 哈希存储；token 为 HMAC-SHA256 签名，有效期 24h（`MIHC_AUTH_TOKEN_TTL`）。

## 6. 常见问题

- **网关不支持 /embeddings**：改为本地嵌入。需给后端镜像补装模型依赖并把模型放进镜像：
  `pip install -r requirements-models.txt`，并将 `models/bge-m3` 放入镜像（或挂载 PVC），
  设置 `EMBEDDING_PROVIDER=local_bge_m3`、`EMBEDDING_DIM=1024`、`MILVUS_DIM=1024`。
- **Milvus 异常**：向量路自动降级，BM25 关键词路仍可回答（检索不中断）。
- **重排未启用**：候选按 RRF 融合分排序，回答仍带引用。
- **数据丢失**：所有可写数据在 `/data`（PVC），重建 Pod 不丢数据；SQLite 为演示回退，
  正式环境可把 `DATABASE_URL` 换成 Sealos PostgreSQL 实例连接串。
- **HTTPS**：Sealos 域名自动带证书；若自定义域名需在 Sealos 网关配置。
