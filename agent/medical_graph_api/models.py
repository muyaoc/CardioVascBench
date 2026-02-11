"""Pydantic 数据模型定义。

本模块集中定义所有 HTTP 接口的请求体/消息体结构，便于：
- FastAPI 自动校验入参
- 生成 OpenAPI 文档
- 与 OpenAI 兼容接口保持一致的数据结构
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .config import CHAT_MODEL_NAME


class QueryRequest(BaseModel):
    question: str
    mode: str = "hybrid"
    enable_rerank: bool = True
    agent_name: str = "anonymous"


class IngestRequest(BaseModel):
    text: str
    source: str = "medical_book"
    agent_name: str = "admin"


class SearchRequest(BaseModel):
    query: str
    top_k: int = 20


class CacheConfigRequest(BaseModel):
    enabled: bool


class OpenAIChatMessage(BaseModel):
    role: str
    content: str


class OpenAICompletionRequest(BaseModel):
    model: str = CHAT_MODEL_NAME
    messages: List[OpenAIChatMessage]
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    response_format: Optional[Dict[str, Any]] = None
