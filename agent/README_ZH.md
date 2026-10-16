# Medical Graph RAG API（Neo4j + 向量检索 + OpenAI 兼容接口）

[English](./README.md)

本项目是一个基于 **FastAPI** 的后端服务，提供“**向量检索 + 知识图谱检索**”的 RAG 能力，并封装为：

- OpenAI 兼容：`POST /v1/chat/completions`
- 自定义检索问答：`POST /api/query`
- 入库与管理：`/api/admin/*`
- 搜索与调试：`/api/search/*`、`/api/debug/*`

默认服务地址：`http://127.0.0.1:8801`     

---

## 1. 项目结构

| 路径 | 作用 |
|---|---|
| `server.py` | 服务启动入口（`uvicorn.run`） |
| `requirements.txt` | Python 依赖 |
| `input.py` | 批量上传 Markdown 到 `/api/admin/ingest` 的脚本 |
| `clean_data/` | 示例/建库数据目录（以仓库为准） |
| `medical_graph_api/` | 核心业务包 |

`medical_graph_api/` 内部文件：

| 文件 | 作用 |
|---|---|
| `app_factory.py` | FastAPI 应用工厂：注册路由、编排检索与生成 |
| `config.py` | 配置（模型、Neo4j、token 等） |
| `generation.py` | 生成回答（诊断文本 / Benchmark JSON） |
| `ingestion.py` | 入库流水线（切分、embedding、写图谱） |
| `lifespan.py` | 启动期初始化（LLM/Embedding/Neo4j/向量索引等） |
| `logging_setup.py` | 日志初始化 |
| `models.py` | Pydantic 请求/响应模型 |
| `retrieval.py` | 检索逻辑（向量检索/图谱检索，异步封装） |
| `security.py` | 鉴权（Bearer Token） |
| `state.py` | 全局状态（LLM/Embedding/Neo4j、job_status 等） |
| `utils.py` | 工具函数（JSON 提取等） |

---

## 2. 环境要求

| 组件 | 版本/要求 |
|---|---|
| Python | 3.9+（建议） |
| Neo4j | 4.x/5.x（向量字段与相关 Cypher 语法需兼容；不同版本细节可能略有差异） |

---

## 3. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---

## 4. 配置说明（重要）

配置集中在 `medical_graph_api/config.py`。

### 4.1 配置项速览

| 类别 | 变量（示例） | 用途 |
|---|---|---|
| 工具模型（入库抽取图谱） | `TOOL_BASE_URL` / `TOOL_API_KEY` / `TOOL_MODEL_NAME` | 实体/关系抽取、图谱构建 |
| 专家模型（问答生成） | `CHAT_BASE_URL` / `CHAT_API_KEY` / `CHAT_MODEL_NAME` | 生成诊断/治疗建议、Benchmark 输出 |
| Rerank | `RERANK_MODEL` | 对向量召回候选进行重排序（如启用） |
| Neo4j | `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` | 图数据库连接 |
| 鉴权 | `SERVER_ACCESS_TOKEN` | API Bearer Token |
| 缓存 | `DEFAULT_CACHE_ENABLE` | 默认缓存开关 |

### 4.2 Neo4j 环境变量覆盖

| 环境变量 | 默认值 |
|---|---|
| `NEO4J_URI` | `bolt://localhost:7689` |
| `NEO4J_USERNAME` | `neo4j` |
| `NEO4J_PASSWORD` | 见 `config.py` |

### 4.3 鉴权

| 项目 | 值 |
|---|---|
| 无需鉴权 | `GET /api/health` |
| 需要鉴权 | 其它绝大部分接口 |
| Header | `Authorization: Bearer <token>` |
| 默认 token | `sk-drug`（来自 `SERVER_ACCESS_TOKEN`） |

### 4.4 安全提醒

| 风险 | 建议 |
|---|---|
| API Key/密码硬编码 | 生产环境迁移到环境变量或密钥管理 |

---

## 5. 启动服务

| 动作 | 命令 |
|---|---|
| 启动 | `python server.py` |
| 健康检查 | `GET http://127.0.0.1:8801/api/health` |

---

## 6. 系统架构与流程图

### 6.1 系统架构（组件关系）

```markdown
+----------------------------------------------------+
|                  客户端 (Client)                   |
| +------------------------------------------------+ |
| |               curl / SDK / 前端                | |
| +------------------------------------------------+ |
+------------------------|---------------------------+
                         | (HTTP + Bearer Token)
+------------------------v---------------------------+
|                FastAPI 服务 (API)                  |
| +------------------------------------------------+ |
| |  [ server.py -> app_factory.py (路由) ]        | |
| |      |                                         | |
| |      +--> [ ingestion.py (入库) ]              | |
| |      +--> [ retrieval.py (检索) ]              | |
| |      +--> [ generation.py (生成) ]             | |
| |                                                | |
| |  [ lifespan.py (初始化) ]                      | |
| +------------------------------------------------+ |
+----------------/--------------------\-------------+
                 /                      \
+---------------v-----------+  +---------v----------+
|  外部依赖 (External)      |  |  外部依赖 (External)   |
| +-----------------------+ |  | +------------------+ |
| | (Neo4j 图数据库)      | |  | | (LLM/Embedding API)| |
| +-----------------------+ |  | +------------------+ |
+---------------------------+  +--------------------+

[入库] -> [写入 Neo4j]
[检索] -> [查询 Neo4j]
[入库/生成/检索] -> [调用 LLM/Embedding API]
```

### 6.2 服务启动初始化流程

对应代码：`medical_graph_api/lifespan.py` → `initialize_components()`

```markdown
用户
  |
  |  python server.py
  v
server.py (uvicorn)
  |
  v
FastAPI create_app() + lifespan.startup
  |
  +--> 初始化 LLM/Embedding/Rerank 客户端
  |
  +--> 建立 Neo4j 连接 / 准备索引
  |
  v
写入全局 state（state.llm_chat / state.llm_tool / state.embeddings / state.graph / state.vector_index ...）
  |
  v
服务启动完成（可调用 /api/health 等接口）
```

### 6.3 端到端流程图（入库与问答）

- 入库端到端：`/api/admin/ingest` → `ingestion.py`
- 问答端到端：`/api/query` 或 `/v1/chat/completions` →（检索）→（生成）

```markdown
[入库流程]
  POST /api/admin/ingest
        |
        v
   文本切分（Markdown 标题切分 + Chunk 切分）
        |
        +--> embedding 生成 -----> 写入 Neo4j Chunk
        |
        +--> Tool LLM 抽取实体/关系 -> 写入 Neo4j 图谱
        |
        v
  GET /api/admin/progress（查看进度）


[问答流程]
  POST /api/query 或 POST /v1/chat/completions
        |
        +--> 向量检索（可选 rerank）
        |
        +--> 图谱检索
        |
        v
   拼接 context
        |
        v
  Chat LLM 生成回答
        |
        v
  返回 answer / choices
```

---

## 7. 入库（构建 Chunk 向量库 + 图谱）

### 7.1 接口与行为

| 项目 | 说明 |
|---|---|
| 接口 | `POST /api/admin/ingest` |
| 触发方式 | 后台任务（`BackgroundTasks`） |
| 进度接口 | `GET /api/admin/progress` |
| 并发限制 | 同一时间只允许一个入库任务；否则 `400 Job running` |
| 主要实现 | `medical_graph_api/ingestion.py` |

### 7.2 入库流程图

```markdown
POST /api/admin/ingest
    |
    v
Markdown 标题切分（#/##/###）
    |
    v
Chunk 切分（chunk_size=800, overlap=100）
    |
    +--> 生成 embedding ------------------> 写入 Neo4j Chunk
    |                                         - 字段: text / embedding / source / header / id
    |
    +--> Tool LLM 抽取实体/关系 -----------> 写入 Neo4j 图谱
    |
    v
刷新 schema
    |
    v
GET /api/admin/progress（查看进度）
```

### 7.3 入库示例

```bash
curl -X POST http://127.0.0.1:8801/api/admin/ingest \
  -H "Authorization: Bearer sk-drug" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "# 示例标题\n这里是一些医学内容...",
    "source": "demo",
    "agent_name": "admin"
  }'
```

---

## 8. Agent 算法与工作流

### 8.1 组件视图

| 组件 | 输入 | 输出 | 主要用途 |
|---|---|---|---|
| 向量检索（Vector） | `question` / `user msg` | 相似 Chunk 文本 | 语义召回证据 |
| 图谱检索（Graph） | `question` / `user msg` | 关系/实体相关文本 | 结构化关系证据 |
| 工具模型（Tool LLM） | 文本 chunk | 实体/关系图结构 | 入库阶段建图 |
| 专家模型（Chat LLM） | `context` + `question` | 最终回答 | 在线问答生成 |

### 8.2 在线问答流程（两条入口）

```markdown
[入口 A] POST /api/query
    |
    v
  判断 mode
    |
    +--> mode = vector  -> 仅向量检索
    |
    +--> mode = graph   -> 仅图谱检索
    |
    +--> mode = hybrid  -> 向量检索 + 图谱检索（并行）
    |
    v
  拼接 context
    |
    v
  Chat LLM 生成 answer
    |
    v
  返回: { answer, sources }


[入口 B] POST /v1/chat/completions
    |
    +--> 向量检索（默认带 rerank）
    |
    +--> 图谱检索
    |
    v
  拼接 context
    |
    v
  判断是否 Benchmark
    |
    +--> response_format.type != json_object  -> 输出诊断+治疗（Markdown）
    |
    +--> response_format.type =  json_object  -> 输出 Benchmark JSON
    |
    v
  返回 OpenAI 兼容结构: { id, choices, usage, ... }
```

---

## 9. 检索问答

### 9.1 `/api/query`

| 字段 | 类型 | 说明 |
|---|---|---|
| `question` | string | 用户问题 |
| `mode` | string | `hybrid` / `vector` / `graph` |
| `enable_rerank` | bool | 是否启用 rerank（由检索实现决定） |
| `agent_name` | string | 调用方标识（用于日志/统计） |

示例：

```bash
curl -X POST http://127.0.0.1:8801/api/query \
  -H "Authorization: Bearer sk-drug" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "高血压有什么症状？",
    "mode": "hybrid",
    "enable_rerank": false,
    "agent_name": "test"
  }'
```

### 9.2 `/v1/chat/completions`

| 项目 | 说明 |
|---|---|
| 接口 | `POST /v1/chat/completions` |
| 默认行为 | 并行向量检索（带 rerank）+ 图谱检索 → 拼接上下文 → 生成 |
| Benchmark 模式 | `response_format: {"type": "json_object"}` 时强制 JSON |

示例：

```bash
curl -X POST http://127.0.0.1:8801/v1/chat/completions \
  -H "Authorization: Bearer sk-drug" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [
      {"role": "user", "content": "高血压怎么诊断？"}
    ]
  }'
```

---

## 10. 其它接口速览

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/health` | 健康检查（无需鉴权） |
| POST | `/api/admin/config/cache` | 开关服务端缓存 |
| GET | `/api/admin/progress` | 入库任务进度 |
| POST | `/api/search/entities` | 实体模糊搜索 |
| POST | `/api/search/relations` | 关系搜索 |
| POST | `/api/search/chunks` | Chunk 相似搜索 |
| GET | `/api/debug/stats` | 图谱/Chunk 统计 |
| GET | `/api/history` | 历史（内存变量） |
| DELETE | `/api/history` | 清空历史与缓存 |

---

## 11. 常见问题

| 问题 | 可能原因 | 处理建议 |
|---|---|---|
| Neo4j 连接失败 | 地址/端口/密码错误 | 检查 `NEO4J_URI`/账号密码、Neo4j 是否运行 |
| 入库返回 400（Job running） | 已有入库任务在跑 | 等待结束后重试；用 `/api/admin/progress` 查看进度 |

---

## 12. 说明

README 以当前代码为准：路由定义在 `medical_graph_api/app_factory.py`，入库逻辑在 `medical_graph_api/ingestion.py`。
