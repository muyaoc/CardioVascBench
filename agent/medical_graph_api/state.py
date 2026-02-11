"""全局状态与任务状态管理。

本模块定义：
- `GlobalState`：保存应用运行时的核心依赖（Neo4jGraph、向量索引、LLM、Embedding 等）
- `state`：全局单例，供各业务模块共享
- `job_status`：入库后台任务的进度/错误信息，供管理接口查询

注意：这是一个进程内内存状态；多进程/多实例部署时需要额外的共享状态方案。
"""

from typing import Any, Dict, List, Optional

from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import DEFAULT_CACHE_ENABLE


class GlobalState:
    graph: Optional[Neo4jGraph] = None
    vector_index: Optional[Neo4jVector] = None
    llm_tool: Optional[ChatOpenAI] = None
    llm_chat: Optional[ChatOpenAI] = None
    llm_format: Optional[ChatOpenAI] = None
    embeddings: Optional[OpenAIEmbeddings] = None
    history: List[Dict] = []
    cache: Dict[str, Any] = {}
    enable_cache: bool = DEFAULT_CACHE_ENABLE

    allowed_nodes = ["Disease", "Drug", "Symptom", "Mechanism", "RiskFactor", "Test"]
    allowed_rels = [
        "TREATS",
        "CONTRAINDICATES",
        "INTERACTS_WITH",
        "CAUSES",
        "HAS_RISK_FACTOR",
    ]


state = GlobalState()

job_status: Dict[str, Any] = {
    "is_running": False,
    "total_batches": 0,
    "current_batch": 0,
    "total_docs": 0,
    "error": None,
}
