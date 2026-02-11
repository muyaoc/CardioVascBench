"""文档入库与图谱构建流水线。

本模块负责将输入的原始文本：
- 按 Markdown 标题层级切分，并进一步按固定长度切分为 Chunk
- 为每个 Chunk 生成向量（embedding），写入 Neo4j 的 `Chunk` 节点
- 调用 LLM 将文本转换为图结构（实体/关系），写入 Neo4j 图数据库

该流程通常由管理接口 `/api/admin/ingest` 以后台任务方式触发。
"""

import asyncio
import gc
import logging
import traceback

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .state import job_status, state

logger = logging.getLogger("MedicalGraphAPI")


async def process_ingestion_pipeline(text: str, source: str, agent: str):
    logger.info(f"📥 用户[{agent}] 开始处理文档: {source}...")
    job_status.update({"is_running": True, "error": None, "current_batch": 0, "total_batches": 0})
    try:
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "Chapter"), ("##", "Section"), ("###", "Subsection")]
        )
        md_splits = splitter.split_text(text)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

        final_splits = []
        for doc in md_splits:
            header = " > ".join(
                [v for k, v in doc.metadata.items() if k in ["Chapter", "Section", "Subsection"]]
            )
            sub_splits = text_splitter.split_text(doc.page_content)
            for sub_content in sub_splits:
                final_splits.append(
                    Document(
                        page_content=f"【{header}】\n{sub_content}" if header else sub_content,
                        metadata={"source": source, "header": header},
                    )
                )

        total = len(final_splits)
        BATCH_SIZE = 3
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        job_status.update({"total_docs": total, "total_batches": total_batches})

        transformer = LLMGraphTransformer(
            llm=state.llm_tool,
            allowed_nodes=state.allowed_nodes,
            allowed_relationships=state.allowed_rels,
            strict_mode=False,
        )

        CONCURRENCY = 3
        semaphore = asyncio.Semaphore(CONCURRENCY)

        async def process_batch_with_retry(batch_idx, batch_docs):
            async with semaphore:
                MAX_RETRIES = 3
                for attempt in range(MAX_RETRIES):
                    try:
                        check_text = batch_docs[0].page_content[:50]
                        check_source = batch_docs[0].metadata.get("source", "")
                        exists = await asyncio.to_thread(
                            state.graph.query,
                            "MATCH (c:Chunk) WHERE c.source = $src AND c.text = $txt RETURN count(c) > 0 as exists",
                            {"src": check_source, "txt": check_text},
                        )
                        if exists and exists[0]["exists"]:
                            if batch_idx % 10 == 0:
                                logger.info(f"⏭️ 批次 {batch_idx}/{total_batches} 已在库中...")
                            job_status["current_batch"] += 1
                            return

                        logger.info(f"🔄 入库批次 {batch_idx}/{total_batches}...")
                        texts = [d.page_content for d in batch_docs]
                        embeddings = await asyncio.to_thread(state.embeddings.embed_documents, texts)

                        params_data = [
                            {
                                "text": d.page_content,
                                "embedding": embeddings[i],
                                "source": d.metadata.get("source", ""),
                                "header": d.metadata.get("header", ""),
                            }
                            for i, d in enumerate(batch_docs)
                        ]

                        await asyncio.to_thread(
                            state.graph.query,
                            "UNWIND $data AS row CREATE (c:Chunk) SET c.text = row.text, c.embedding = row.embedding, c.source = row.source, c.header = row.header, c.id = randomUUID()",
                            {"data": params_data},
                        )

                        graph_docs = await asyncio.to_thread(
                            transformer.convert_to_graph_documents, batch_docs
                        )
                        if graph_docs:
                            await asyncio.to_thread(state.graph.add_graph_documents, graph_docs)

                        job_status["current_batch"] += 1
                        if batch_idx % 20 == 0:
                            gc.collect()
                        return

                    except Exception as e:
                        logger.warning(f"⚠️ 批次 {batch_idx} 异常: {e}")
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(2 * (attempt + 1))
                        else:
                            logger.error(f"❌ 批次 {batch_idx} 失败")
                            job_status["current_batch"] += 1

        tasks = []
        for i in range(0, total, BATCH_SIZE):
            tasks.append(process_batch_with_retry((i // BATCH_SIZE) + 1, final_splits[i : i + BATCH_SIZE]))

        await asyncio.gather(*tasks)
        state.graph.refresh_schema()
        logger.info(f"🎉 文件 [{source}] 处理完成！")

    except Exception:
        logger.error(f"Ingest Error: {traceback.format_exc()}")
        job_status["error"] = str(traceback.format_exc())
    finally:
        job_status["is_running"] = False
