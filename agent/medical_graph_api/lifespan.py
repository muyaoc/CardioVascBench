"""应用启动初始化（lifespan）相关逻辑。

本模块负责在 FastAPI 启动阶段初始化核心依赖：
- Embedding 模型（用于向量化）
- 工具 LLM（用于实体提取/图谱构建）
- 专家 LLM（用于回答/诊断/Benchmark JSON 生成）
- Neo4j 连接与索引检查
- Neo4jVector 向量索引连接

注意：此处的初始化会在应用启动时执行一次；如果初始化失败，服务将无法正常启动。
"""

import logging

from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import (
    CHAT_API_KEY,
    CHAT_BASE_URL,
    CHAT_MODEL_NAME,
    FORMAT_API_KEY,
    FORMAT_BASE_URL,
    FORMAT_MODEL_NAME,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    TOOL_API_KEY,
    TOOL_BASE_URL,
    TOOL_MODEL_NAME,
)
from .state import state

logger = logging.getLogger("MedicalGraphAPI")


def create_db_indexes():
    if not state.graph:
        return
    try:
        state.graph.query(
            """
        CREATE VECTOR INDEX medical_chunks IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS {indexConfig: {
          `vector.dimensions`: 1024,
          `vector.similarity_function`: 'cosine'
        }}
        """
        )
        try:
            state.graph.query(
                "CREATE INDEX chunk_source_index IF NOT EXISTS FOR (c:Chunk) ON (c.source)"
            )
            logger.info("✅ 数据库索引检查完毕")
        except Exception:
            pass
    except Exception:
        pass


def initialize_components():
    logger.info("🚀 系统初始化...")
    try:
        # Embedding
        state.embeddings = OpenAIEmbeddings(
            model="BAAI/bge-m3",
            openai_api_key=TOOL_API_KEY,
            openai_api_base=TOOL_BASE_URL,
            check_embedding_ctx_length=False,
            timeout=30,
            max_retries=2,
        )

        # 1. 工具模型 (SiliconFlow)
        state.llm_tool = ChatOpenAI(
            model=TOOL_MODEL_NAME,
            openai_api_key=TOOL_API_KEY,
            openai_api_base=TOOL_BASE_URL,
            temperature=0.0,
            max_retries=3,
            request_timeout=60,
        )

        # 2. 医学模型（临床推理/候选诊断生成）
        # 🚀 关键修复：设置 600秒超时，防止 502/超时错误
        state.llm_chat = ChatOpenAI(
            model=CHAT_MODEL_NAME,
            openai_api_key=CHAT_API_KEY,
            openai_api_base=CHAT_BASE_URL,
            temperature=0.1,
            max_retries=3,
            request_timeout=600,  # 👈 必须足够长，等待推理完成
        )

        # 3. 通用模型（抽取/归一/输出严格JSON）
        state.llm_format = ChatOpenAI(
            model=FORMAT_MODEL_NAME,
            openai_api_key=FORMAT_API_KEY,
            openai_api_base=FORMAT_BASE_URL,
            temperature=0.0,
            max_retries=3,
            request_timeout=120,
        )

        state.graph = Neo4jGraph(
            url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD
        )
        create_db_indexes()

        try:
            state.vector_index = Neo4jVector.from_existing_graph(
                embedding=state.embeddings,
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                index_name="medical_chunks",
                node_label="Chunk",
                text_node_properties=["text", "header", "source"],
                embedding_node_property="embedding",
            )
            logger.info("✅ 向量索引连接成功")
        except Exception:
            logger.info("ℹ️ 向量索引暂无数据")

    except Exception as e:
        logger.error(f"❌ 初始化失败: {e}")
        raise e
