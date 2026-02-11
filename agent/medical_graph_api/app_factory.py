"""FastAPI 应用工厂模块。

本模块负责：
- 定义应用生命周期（lifespan），在启动时初始化 LLM/Embedding/Neo4j 等组件
- 注册所有对外 HTTP API 路由（OpenAI 兼容接口、入库、检索、搜索、调试等）

设计为工厂函数 `create_app()`，方便在不同运行方式（脚本启动/ASGI 部署/测试）下复用。

注意：本模块只做路由装配，不应包含重业务逻辑；业务逻辑在 generation/ingestion/retrieval 等模块中。
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException

from .generation import generate_benchmark_json
from .ingestion import process_ingestion_pipeline
from .lifespan import initialize_components
from .models import (
    CacheConfigRequest,
    IngestRequest,
    OpenAICompletionRequest,
    QueryRequest,
    SearchRequest,
)
from .retrieval import async_graph_search, async_vector_search
from .security import verify_token
from .state import job_status, state
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_components()
    yield


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    @app.post("/v1/chat/completions", dependencies=[Depends(verify_token)])
    async def openai_chat_completions(req: OpenAICompletionRequest):
        start_time = time.time()
        last_message = req.messages[-1].content

        # 1. 并发检索 (带超时控制)
        tasks = [async_vector_search(last_message, True), async_graph_search(last_message)]
        results = await asyncio.gather(*tasks)
        context = (results[0] or "") + "\n" + (results[1] or "")

        # 2. Benchmark-only：统一走两阶段诊断 JSON
        answer_text = await generate_benchmark_json(context, req.messages)

        duration = time.time() - start_time

        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(last_message),
                "completion_tokens": len(answer_text),
                "total_tokens": 0,
            },
            "system_fingerprint": f"latency_{duration:.3f}s",
        }

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/admin/config/cache", dependencies=[Depends(verify_token)])
    def set_cache_config(config: CacheConfigRequest):
        state.enable_cache = config.enabled
        return {"status": "success", "cache_enabled": state.enable_cache}

    @app.post("/api/admin/ingest", dependencies=[Depends(verify_token)])
    def ingest(req: IngestRequest, bg: BackgroundTasks):
        if job_status["is_running"]:
            raise HTTPException(400, "Job running")
        bg.add_task(process_ingestion_pipeline, req.text, req.source, req.agent_name)
        return {"status": "started", "msg": f"Resuming by {req.agent_name}"}

    @app.get("/api/admin/progress", dependencies=[Depends(verify_token)])
    def get_progress():
        return job_status

    @app.post("/api/query", dependencies=[Depends(verify_token)])
    async def query(req: QueryRequest):
        tasks = [
            async_vector_search(req.question, req.enable_rerank)
            if req.mode in ["hybrid", "vector"]
            else asyncio.sleep(0),
            async_graph_search(req.question)
            if req.mode in ["hybrid", "graph"]
            else asyncio.sleep(0),
        ]
        results = await asyncio.gather(*tasks)
        context = (results[0] or "") + "\n" + (results[1] or "")

        prompt = ChatPromptTemplate.from_template(
            """基于本地知识库回答。无数据则声明并补充。\n本地知识库:\n{context}\n问题: {question}"""
        )
        chain = prompt | state.llm_chat | StrOutputParser()
        answer = await asyncio.to_thread(
            chain.invoke, {"context": context, "question": req.question}
        )

        return {"id": str(uuid.uuid4()), "answer": answer, "sources": context}

    @app.post("/api/search/entities", dependencies=[Depends(verify_token)])
    def search_entities(req: SearchRequest):
        if not state.graph:
            raise HTTPException(503, "No DB")
        try:
            data = state.graph.query(
                "MATCH (n) WHERE toLower(n.id) CONTAINS toLower($q) RETURN n.id, labels(n), properties(n) LIMIT $k",
                params={"q": req.query, "k": req.top_k},
            )
            return {
                "results": [
                    {
                        "name": i["n.id"],
                        "type": i["labels(n)"][0],
                        "props": str(i["properties(n)"]),
                    }
                    for i in data
                ]
            }
        except Exception as e:
            return {"error": str(e)}

    @app.post("/api/search/relations", dependencies=[Depends(verify_token)])
    def search_relations(req: SearchRequest):
        if not state.graph:
            return {"count": 0, "results": []}
        try:
            data = state.graph.query(
                "MATCH (s)-[r]->(t) WHERE type(r) CONTAINS toUpper($q) OR toLower(s.id) CONTAINS toLower($q) OR toLower(t.id) CONTAINS toLower($q) RETURN s.id as source, type(r) as relation, t.id as target LIMIT $k",
                params={"q": req.query, "k": req.top_k},
            )
            return {"query": req.query, "count": len(data), "results": data}
        except Exception as e:
            return {"query": req.query, "count": 0, "results": [], "error": str(e)}

    @app.post("/api/search/chunks", dependencies=[Depends(verify_token)])
    def search_chunks(req: SearchRequest):
        if not state.vector_index:
            return {"count": 0, "results": []}
        try:
            docs_and_scores = state.vector_index.similarity_search_with_score(
                req.query, k=req.top_k
            )
            results = [
                {
                    "content": d.page_content,
                    "source": d.metadata.get("source", ""),
                    "score": s,
                }
                for d, s in docs_and_scores
            ]
            return {"query": req.query, "count": len(results), "results": results}
        except Exception as e:
            return {"query": req.query, "count": 0, "results": [], "error": str(e)}

    @app.get("/api/history", dependencies=[Depends(verify_token)])
    def get_history(limit: int = 10):
        return {"history": state.history[:limit]}

    @app.delete("/api/history", dependencies=[Depends(verify_token)])
    def clear_history():
        state.history = []
        state.cache = {}
        return {"status": "cleared"}

    @app.get("/api/debug/stats", dependencies=[Depends(verify_token)])
    def stats():
        if not state.graph:
            return {"error": "No DB"}
        try:
            nodes = state.graph.query("MATCH (n) RETURN labels(n) as l, count(*) as c")
            chunks = state.graph.query("MATCH (n:Chunk) RETURN count(n) as count")
            return {"nodes_distribution": nodes, "vector_chunks": chunks}
        except Exception as e:
            return {"error": str(e)}

    return app
