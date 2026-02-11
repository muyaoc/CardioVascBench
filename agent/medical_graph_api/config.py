"""应用程序的配置模块。

集中管理所有硬编码的配置变量，例如 API 密钥、模型名称、数据库连接信息等。
建议将敏感信息（如 API Key）迁移至环境变量或专门的密钥管理服务。
"""

import os

# --- 1. 工具模型 (实体提取/图谱构建) ---
TOOL_BASE_URL = "https://api.siliconflow.cn/v1"
TOOL_API_KEY = "sk-xxx"
TOOL_MODEL_NAME = "Qwen/Qwen3-8B"

# --- 2. 专家/医学模型 (临床诊断/候选诊断生成) ---
CHAT_BASE_URL = "http://223.109.239.65:31962/v1"
CHAT_API_KEY = "sk-xxx"
CHAT_MODEL_NAME = "OpenBioLLM-70B"

# --- 3. 通用模型 (抽取/归一/格式化) ---
NORMAL_BASE_URL = "https://aihubmix.com/v1"
NORMAL_API_KEY = "sk-xxx"
NORMAL_MODEL_NAME = "gpt-5.2"


# --- 4. Rerank 模型 ---
RERANK_BASE_URL = "https://api.siliconflow.cn/v1"
RERANK_API_KEY = "sk-xxx"
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# 方案实现用：显式区分两阶段用到的模型（默认复用上面的配置）
DIAGNOSE_BASE_URL = CHAT_BASE_URL
DIAGNOSE_API_KEY = CHAT_API_KEY
DIAGNOSE_MODEL_NAME = CHAT_MODEL_NAME

FORMAT_BASE_URL = NORMAL_BASE_URL
FORMAT_API_KEY = NORMAL_API_KEY
FORMAT_MODEL_NAME = NORMAL_MODEL_NAME

# --- Benchmark 专用：多阶段后处理开关与目标输出控制 ---
# 是否启用“LLM 后处理增强”（同义归一/更贴近评测集命名/主诊断校准等）
ENABLE_LLM_POSTPROCESS = True

# 最终输出诊断条数目标（软约束，仍然允许 1..10）
BENCHMARK_TARGET_MIN_DIAGNOSES = 1
BENCHMARK_TARGET_MAX_DIAGNOSES = 2

# 是否启用“医学模型主诊断校准”阶段（在通用模型抽取后再校准 Top1）
ENABLE_PRIMARY_CALIBRATION = True

# Rerank 模型
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# Neo4j 配置
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7689")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv(
    "NEO4J_PASSWORD",
    "xxx",
)

SERVER_ACCESS_TOKEN = "sk-xxx"
DEFAULT_CACHE_ENABLE = True

# --- Benchmark 常用心脑血管诊断标准名称库 ---
BENCHMARK_COMMON_DIAGNOSES = [
    "冠状动脉性心脏病", "胸痛", "胸闷", "冠状动脉粥样硬化性心脏病", "不稳定性心绞痛",
    "高血压3级", "晕厥", "心悸", "高血压病3级(极高危)", "心律失常", "头晕", "高血压",
    "慢性心功能不全急性加重", "心房颤动", "慢性心力衰竭", "阵发性心房颤动", "高脂血症",
    "心力衰竭", "高血压病3级(高危)", "冠状动脉粥样硬化", "频发性室性期前收缩",
    "急性心力衰竭", "高血压2级", "阵发性室上性心动过速", "2型糖尿病", "稳定性心绞痛",
    "心功能I级(NYHA分级)", "室性期前收缩", "呼吸困难", "高血压病2级(高危)",
    "预激综合征", "心包积液", "持续性心房颤动", "背痛", "高血压急症",
    "高血压病2级(极高危)", "心脏扩大", "心房扑动", "乏力", "房性期前收缩",
    "肥厚型心肌病", "窦性心动过缓", "陈旧性心肌梗死"
]
