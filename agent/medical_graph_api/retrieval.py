"""检索相关逻辑（向量检索 + 图谱检索）。

本模块提供两类检索能力：
- 向量检索：基于 Neo4jVector 的相似度搜索，可选 rerank 提升相关性
- 图谱检索：先用 LLM 从问题中提取实体，再在 Neo4j 中查询相关关系

返回值统一为字符串形式的“可拼接上下文”，供生成模块用于 RAG。
"""

import asyncio
import logging

import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .config import RERANK_API_KEY, RERANK_BASE_URL, RERANK_MODEL_NAME
from .state import state

logger = logging.getLogger("MedicalGraphAPI")


async def async_vector_search(question: str, enable_rerank: bool) -> str:
    if not state.vector_index:
        return ""
    try:
        top_k = 50 if enable_rerank else 5
        docs = await asyncio.to_thread(state.vector_index.similarity_search, question, k=top_k)
        if not docs:
            return ""

        final_docs = docs
        if enable_rerank:
            try:
                headers = {
                    "Authorization": f"Bearer {RERANK_API_KEY}",
                    "Content-Type": "application/json",
                }
                query_truncated = question[:1000]
                payload = {
                    "model": RERANK_MODEL_NAME,
                    "query": query_truncated,
                    "documents": [d.page_content for d in docs],
                    "top_n": 8,
                    "return_documents": False,
                }

                def call_rerank():
                    return requests.post(
                        f"{RERANK_BASE_URL}/rerank",
                        json=payload,
                        headers=headers,
                        timeout=20,
                    )

                resp = await asyncio.to_thread(call_rerank)
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    reranked = [
                        docs[item["index"]]
                        for item in results
                        if item["relevance_score"] > 0.01
                    ]
                    final_docs = reranked if reranked else docs[:5]
            except Exception:
                final_docs = docs[:5]

        res = "\n【📚 文档片段】:\n"
        for idx, d in enumerate(final_docs):
            res += (
                f"[{idx+1}] (来源: {d.metadata.get('source', '')})\n"
                f"内容: {d.page_content[:500]}...\n"
            )
        return res
    except Exception as e:
        logger.error(f"Vector Error: {e}")
        return ""


async def async_graph_search(question: str) -> str:
    if not state.graph:
        return ""
    try:
        async def _do_search():
            short_question = question[:800]
            prompt = ChatPromptTemplate.from_template(
                "从以下文本中提取关键医学实体（疾病、症状），最多不超过5个，忽略无关修饰词。仅输出实体名称，用逗号分隔，不要包含其他内容。\n文本: {question}"
            )
            chain = prompt | state.llm_tool | StrOutputParser()
            entities_str = await asyncio.to_thread(chain.invoke, {"question": short_question})
            entities = [
                e.strip()
                for e in entities_str.split(",")
                if e.strip() and len(e.strip()) > 1
            ]
            if not entities:
                return ""
            logger.info(f"🔍 [Graph] 提取实体: {entities}")

            template = """
            MATCH (n)-[r]-(m)
            WHERE toLower(n.id) IN $names OR any(name IN $names WHERE toLower(n.id) CONTAINS toLower(name))
            RETURN n.id AS source, type(r) AS rel, m.id AS target
            LIMIT 30
            """
            res = await asyncio.to_thread(
                state.graph.query, template, params={"names": entities}
            )
            if res:
                text = "\n【🕸️ 图谱数据】:\n"
                seen = set()
                for item in res:
                    relation_str = (
                        f"({item['source']}) --[{item['rel']}]--> ({item['target']})"
                    )
                    if relation_str not in seen:
                        text += relation_str + "\n"
                        seen.add(relation_str)
                return text
            return ""

        return await asyncio.wait_for(_do_search(), timeout=15.0)
    except asyncio.TimeoutError:
        logger.warning("⚠️ 图谱检索超时(15s)，跳过。")
        return ""
    except Exception as e:
        logger.error(f"Graph Error: {e}")
        return ""
