# Medical Graph RAG API (Neo4j + Vector Retrieval + OpenAI-Compatible API)

[中文版](./README_ZH.md)

This **FastAPI** backend provides RAG capabilities through **vector retrieval + knowledge graph retrieval**, exposed through the following APIs:

- OpenAI-compatible chat: `POST /v1/chat/completions`
- Custom retrieval and question answering: `POST /api/query`
- Ingestion and administration: `/api/admin/*`
- Search and debugging: `/api/search/*`, `/api/debug/*`

Default service URL: `http://127.0.0.1:8801`

---

## 1. Project Structure

| Path | Purpose |
|---|---|
| `server.py` | Service entry point (`uvicorn.run`) |
| `requirements.txt` | Python dependencies |
| `input.py` | Script for batch uploading Markdown to `/api/admin/ingest` |
| `clean_data/` | Example data / graph-building data directory (refer to the repository contents) |
| `medical_graph_api/` | Core application package |

Files inside `medical_graph_api/`:

| File | Purpose |
|---|---|
| `app_factory.py` | FastAPI application factory: registers routes and orchestrates retrieval and generation |
| `config.py` | Configuration for models, Neo4j, tokens, etc. |
| `generation.py` | Answer generation (diagnostic text / benchmark JSON) |
| `ingestion.py` | Ingestion pipeline (splitting, embeddings, and graph writes) |
| `lifespan.py` | Startup initialization (LLMs, embeddings, Neo4j, vector index, etc.) |
| `logging_setup.py` | Logging initialization |
| `models.py` | Pydantic request and response models |
| `retrieval.py` | Retrieval logic (vector and graph retrieval, with asynchronous wrappers) |
| `security.py` | Authentication (Bearer token) |
| `state.py` | Global state (LLMs, embeddings, Neo4j, `job_status`, etc.) |
| `utils.py` | Utility functions (JSON extraction, etc.) |

---

## 2. Requirements

| Component | Version / Requirement |
|---|---|
| Python | 3.9+ (recommended) |
| Neo4j | 4.x/5.x (vector fields and the relevant Cypher syntax must be compatible; details may vary by version) |

---

## 3. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---

## 4. Configuration (Important)

Configuration is centralized in `medical_graph_api/config.py`.

### 4.1 Configuration Overview

| Category | Variables (Examples) | Purpose |
|---|---|---|
| Tool model (graph extraction during ingestion) | `TOOL_BASE_URL` / `TOOL_API_KEY` / `TOOL_MODEL_NAME` | Entity and relationship extraction, graph construction |
| Expert model (answer generation) | `CHAT_BASE_URL` / `CHAT_API_KEY` / `CHAT_MODEL_NAME` | Diagnostic and treatment recommendations, benchmark output |
| Reranking | `RERANK_MODEL` | Rerank candidates retrieved by vector search, if enabled |
| Neo4j | `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` | Graph database connection |
| Authentication | `SERVER_ACCESS_TOKEN` | API Bearer token |
| Cache | `DEFAULT_CACHE_ENABLE` | Default cache setting |

### 4.2 Neo4j Environment Variable Overrides

| Environment Variable | Default |
|---|---|
| `NEO4J_URI` | `bolt://localhost:7689` |
| `NEO4J_USERNAME` | `neo4j` |
| `NEO4J_PASSWORD` | See `config.py` |

### 4.3 Authentication

| Item | Value |
|---|---|
| No authentication required | `GET /api/health` |
| Authentication required | Most other endpoints |
| Header | `Authorization: Bearer <token>` |
| Default token | `sk-drug` (from `SERVER_ACCESS_TOKEN`) |

### 4.4 Security Note

| Risk | Recommendation |
|---|---|
| Hardcoded API keys / passwords | Use environment variables or secret management in production |

---

## 5. Start the Service

| Action | Command |
|---|---|
| Start | `python server.py` |
| Health check | `GET http://127.0.0.1:8801/api/health` |

---

## 6. System Architecture and Workflows

### 6.1 System Architecture (Component Relationships)

```text
+----------------------------------------------------+
|                       Client                       |
| +------------------------------------------------+ |
| |              curl / SDK / frontend             | |
| +------------------------------------------------+ |
+------------------------|---------------------------+
                         | (HTTP + Bearer Token)
+------------------------v---------------------------+
|                   FastAPI Service                  |
| +------------------------------------------------+ |
| |  [ server.py -> app_factory.py (routes) ]        | |
| |      |                                         | |
| |      +--> [ ingestion.py (ingestion) ]          | |
| |      +--> [ retrieval.py (retrieval) ]          | |
| |      +--> [ generation.py (generation) ]        | |
| |                                                | |
| |  [ lifespan.py (initialization) ]               | |
| +------------------------------------------------+ |
+----------------/--------------------\--------------+
                 /                    \
+---------------v-----------+  +-------v------------+
| External dependency       |  | External dependency|
| +-----------------------+ |  | +----------------+ |
| | Neo4j graph database  | |  | | LLM/Embedding  | |
| |                       | |  | | API            | |
| +-----------------------+ |  | +----------------+ |
+---------------------------+  +--------------------+

[Ingestion] -> [Write to Neo4j]
[Retrieval] -> [Query Neo4j]
[Ingestion/Generation/Retrieval] -> [Call LLM/Embedding API]
```

### 6.2 Service Startup Initialization

Relevant code: `medical_graph_api/lifespan.py` → `initialize_components()`

```text
User
  |
  |  python server.py
  v
server.py (uvicorn)
  |
  v
FastAPI create_app() + lifespan.startup
  |
  +--> Initialize LLM/Embedding/Rerank clients
  |
  +--> Connect to Neo4j / prepare indexes
  |
  v
Populate global state (state.llm_chat / state.llm_tool / state.embeddings / state.graph / state.vector_index ...)
  |
  v
Service ready (endpoints such as /api/health are available)
```

### 6.3 End-to-End Workflows (Ingestion and Question Answering)

- Ingestion: `/api/admin/ingest` → `ingestion.py`
- Question answering: `/api/query` or `/v1/chat/completions` → retrieval → generation

```text
[Ingestion workflow]
  POST /api/admin/ingest
        |
        v
   Split text (Markdown headings + chunks)
        |
        +--> Generate embeddings -----> Write Neo4j chunks
        |
        +--> Tool LLM extracts entities/relationships -> Write Neo4j graph
        |
        v
  GET /api/admin/progress (check progress)


[Question answering workflow]
  POST /api/query or POST /v1/chat/completions
        |
        +--> Vector retrieval (optional reranking)
        |
        +--> Graph retrieval
        |
        v
   Assemble context
        |
        v
  Chat LLM generates an answer
        |
        v
  Return answer / choices
```

---

## 7. Ingestion (Build the Chunk Vector Store and Graph)

### 7.1 Endpoint and Behavior

| Item | Description |
|---|---|
| Endpoint | `POST /api/admin/ingest` |
| Execution | Background task (`BackgroundTasks`) |
| Progress endpoint | `GET /api/admin/progress` |
| Concurrency limit | Only one ingestion job at a time; otherwise returns `400 Job running` |
| Main implementation | `medical_graph_api/ingestion.py` |

### 7.2 Ingestion Flowchart

```text
POST /api/admin/ingest
    |
    v
Split by Markdown headings (#/##/###)
    |
    v
Split into chunks (chunk_size=800, overlap=100)
    |
    +--> Generate embeddings -----------------> Write Neo4j chunks
    |                                            - Fields: text / embedding / source / header / id
    |
    +--> Tool LLM extracts entities/relationships -> Write Neo4j graph
    |
    v
Refresh schema
    |
    v
GET /api/admin/progress (check progress)
```

### 7.3 Ingestion Example

```bash
curl -X POST http://127.0.0.1:8801/api/admin/ingest \
  -H "Authorization: Bearer sk-drug" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "# Example Heading\nSome medical content...",
    "source": "demo",
    "agent_name": "admin"
  }'
```

---

## 8. Agent Algorithm and Workflow

### 8.1 Component Overview

| Component | Input | Output | Main Purpose |
|---|---|---|---|
| Vector retrieval | `question` / user message | Similar chunk text | Retrieve semantic evidence |
| Graph retrieval | `question` / user message | Relationship/entity-related text | Retrieve structured relationship evidence |
| Tool LLM | Text chunks | Entity/relationship graph structures | Build the graph during ingestion |
| Chat LLM (expert model) | `context` + `question` | Final answer | Generate answers to online queries |

### 8.2 Online Question Answering (Two Entry Points)

```text
[Entry A] POST /api/query
    |
    v
  Check mode
    |
    +--> mode = vector  -> Vector retrieval only
    |
    +--> mode = graph   -> Graph retrieval only
    |
    +--> mode = hybrid  -> Vector + graph retrieval (in parallel)
    |
    v
  Assemble context
    |
    v
  Chat LLM generates answer
    |
    v
  Return: { answer, sources }


[Entry B] POST /v1/chat/completions
    |
    +--> Vector retrieval (reranking enabled by default)
    |
    +--> Graph retrieval
    |
    v
  Assemble context
    |
    v
  Check benchmark mode
    |
    +--> response_format.type != json_object  -> Diagnosis + treatment (Markdown)
    |
    +--> response_format.type =  json_object  -> Benchmark JSON
    |
    v
  Return OpenAI-compatible response: { id, choices, usage, ... }
```

---

## 9. Retrieval and Question Answering

### 9.1 `/api/query`

| Field | Type | Description |
|---|---|---|
| `question` | string | User question |
| `mode` | string | `hybrid` / `vector` / `graph` |
| `enable_rerank` | bool | Whether to enable reranking (subject to the retrieval implementation) |
| `agent_name` | string | Caller identifier (for logging/statistics) |

Example:

```bash
curl -X POST http://127.0.0.1:8801/api/query \
  -H "Authorization: Bearer sk-drug" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the symptoms of hypertension?",
    "mode": "hybrid",
    "enable_rerank": false,
    "agent_name": "test"
  }'
```

### 9.2 `/v1/chat/completions`

| Item | Description |
|---|---|
| Endpoint | `POST /v1/chat/completions` |
| Default behavior | Parallel vector retrieval (with reranking) + graph retrieval → assemble context → generate answer |
| Benchmark mode | Enforces JSON when `response_format: {"type": "json_object"}` is set |

Example:

```bash
curl -X POST http://127.0.0.1:8801/v1/chat/completions \
  -H "Authorization: Bearer sk-drug" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [
      {"role": "user", "content": "How is hypertension diagnosed?"}
    ]
  }'
```

---

## 10. Other Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Health check (no authentication required) |
| POST | `/api/admin/config/cache` | Toggle server-side caching |
| GET | `/api/admin/progress` | Ingestion job progress |
| POST | `/api/search/entities` | Fuzzy entity search |
| POST | `/api/search/relations` | Relationship search |
| POST | `/api/search/chunks` | Chunk similarity search |
| GET | `/api/debug/stats` | Graph/chunk statistics |
| GET | `/api/history` | History (stored in memory) |
| DELETE | `/api/history` | Clear history and cache |

---

## 11. Troubleshooting

| Problem | Possible Cause | Suggested Action |
|---|---|---|
| Neo4j connection failure | Incorrect address, port, or password | Check `NEO4J_URI`, credentials, and whether Neo4j is running |
| Ingestion returns 400 (Job running) | An ingestion job is already running | Wait for completion and retry; check `/api/admin/progress` |

---

## 12. Notes

The current implementation is the source of truth for this README: routes are defined in `medical_graph_api/app_factory.py`, and ingestion logic is in `medical_graph_api/ingestion.py`.
