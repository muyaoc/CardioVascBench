# CardioVascBench 心脑血管疾病大模型评测基准

[English Version (README_EN.md)](./README_EN.md)

面向心脑血管疾病场景的中文评测基准。依据真实心内科数据构建评估集，评估待测大语言模型的“最终诊断”能力与一致性。

**数据下载 (Data Download)**：[CardioVascBench 数据集 (OpenI 启智社区)](https://openi.pcl.ac.cn/explore/datasets/cindyc/CardioVascBench)

本 README 旨在为**用户（尤其是临床医生）**提供快速了解和操作本评测基准的指南。

---

## 1. 目录结构与角色说明

```text
CARDIOVASCBENCH/
├── config/
│   ├── paths.yml                 # 所有路径信息
│   └── weights.yml               # 指标权重信息
├── data/
│   ├── source_data/              # 真实心内科数据与转换脚本
│   ├── llm_eval_data/            # 基准评估数据（eval/test）
│   ├── llms_answers_data_generator.py # 生成模型答案的脚本
│   └── run_answers.sh
├── src/
│   ├── core_evaluator.py         # 核心评测算法（归一化 / 匹配 / 各类指标）
│   └── evaluate_llm.py           # 评估待测 LLM（读取配置，脚本调度）
├── agent/                        # 图谱 RAG Agent 模块 (FastAPI 服务)
├── requirements.txt              # 项目依赖
└── results/
    └── llms_evaluation_results/  # [输出] LLM 评测结果 (JSON / PNG / CSV)
    
```

---

## 2. 环境准备与依赖

### 2.1 Python 与依赖

- 推荐 Python 版本：`>= 3.9`。
- 安装项目依赖：

```bash
pip install -r requirements.txt
```
### 2.2 配置文件总览

所有路径、权重与模型相关参数**必须**从 `config/` 读取，不允许在代码中硬编码：

- `config/paths.yml`：数据 / 答案 / 结果路径；
- `config/weights.yml`：各指标权重与总分系数；
- `config/eval_llm_api.yml`：API 或模型名称等元信息（评估阶段只用于记录）。

---

## 3. 评测流程与运行

评测分为以下三步：

1.  **构建评估集**：从原始病历数据生成用于评估的 JSON 文件。
2.  **生成模型答案**：调用大模型，根据评估集样本生成诊断预测结果。
3.  **离线评估**：运行评估脚本，计算模型得分并生成报告。

### 3.1 快速开始（已有人生成好答案的场景）

如果你已经有了某个模型生成的答案，可以直接运行评估脚本：

```bash
# 对单个模型进行评估
python src/evaluate_llm.py --models your-model-name

# 或一次评估多个模型（英文逗号分隔）
python src/evaluate_llm.py --models Apollo-72B,qwen3
```

脚本会自动读取 `config/paths.yml` 和 `config/weights.yml` 中的配置。

### 3.2 只生成一个模型的答案（直接运行 Python 脚本）

如果你不使用 `run_answers.sh`，而是直接运行：

```bash
API_URL=<placeholder> API_KEY=<placeholder> \
python data/llms_answers_data_generator.py \
  --model-cfg config/llm_tpl_config/<model>.yml \
  --paths-cfg config/paths.yml
```

你需要准备与修改的内容如下：

1.  **环境变量**（必需）  
    - `API_URL`：OpenAI 兼容服务的基础地址。  
    - `API_KEY`：访问密钥。  

2.  **模型配置**（必需）  
    - 使用 `--model-cfg` 指向单个模型的 YAML，例如：
      - `config/llm_tpl_config/qwen2.5-7b-instruct.yml`
    - 该文件决定模型的名称、API 类型和调用参数。

3.  **路径配置**（必需）  
    - 使用 `--paths-cfg` 指向 `config/paths.yml`。  
    - 确保评测输入数据和输出答案的路径配置正确。  

> 注意：`paths.yml` 中的路径若为相对路径，需在项目根目录执行命令，确保相对路径生效。

---

## 4. 评估结果解读

评测结果主要包括以下几类文件：

-   **JSON 报告**：`results/llms_evaluation_results/llm_evaluation_metrics_<model>-<strategy>.json`
    -   包含 `summary_scores.Score_total`（总分）以及详细的 `ranked_multi_diag`（多诊断排序指标，主要参考）和 `combined`（兼容旧版指标，仅供参考）结果。
-   **图表**：`results/llms_evaluation_results/figures_<model>-<strategy>/*.png`
    -   例如 `topk_accuracy_bar_*.png`，展示 Top-k 准确率。
-   **调试 CSV**：`results/llms_evaluation_results/debug_evaluation_<model>-<strategy>_*.csv`
    -   提供每条样本的详细匹配得分和标签，用于深入分析。

### 4.1 综合评分 `Score_total`

`Score_total` 是模型的最终综合得分，分数越高，模型表现越好。该分数在计算时会**强调诊断的排序质量**，这与临床医生在实际诊断中优先给出主诊断、再列出次要诊断的思路高度一致。

---

## 5. 指标定义 (Ranked Multi-Diagnosis & Best-Match)

### 5.1 多诊断排序指标 (Ranked Multi-Diagnosis)

当一个病例存在多个正确诊断（如：房颤、心衰、高血压）时，我们使用信息检索领域的标准指标来评估模型生成的诊断列表。

#### 5.1.1 nDCG@K (归一化折损累计增益)
衡量排序质量。不仅要求模型预测正确，还要求将正确的诊断排在前面。

- **Gain (增益)**: `rel_score * _gt_exp_weight(gt_rank)`
  - `rel_score`: 预测值与金标准的语义相似度（0~1）。
  - `_gt_exp_weight(rank)`: GT 重要性权重，采用指数衰减。其中 rank 为金标准列表中的排名（从1开始），越靠前的主诊断权重越大。
- **Discount (折损)**: `_pos_discount(pred_pos)`
  - `_pos_discount(pos)`: 预测位置折扣，采用标准 DCG 折扣。模型把正确答案排得越靠后，该分值越低。

#### 5.1.2 PrimaryHit@K
衡量模型是否命中了“第一主诊断”。

#### 5.1.3 Precision@K & Recall@K
- **Precision@K**: 前 K 个预测中有多少是正确的。
- **Recall@K**: 金标准列表中的诊断有多少被召回了。

#### 5.1.4 Final Score (综合分)
结合排序质量、主诊断命中率和精确率的综合打分：

### 5.2 单诊断最佳匹配指标 (Best-Match / Combined)

仅关注模型是否“沾边”金标准中的任何一个诊断（通常用于兼容旧指标）。

- **Top-k Accuracy**: 前 k 个预测中是否包含正确答案。
- **MRR (Mean Reciprocal Rank)**: 第一个正确答案排名的倒数。
- **Weighted Semantic Mean**: 所有预测的加权语义相似度均值。

---

## 6. Medical Graph RAG Agent

该模块提供基于 **Neo4j + 向量检索** 的医学知识图谱 RAG 能力，支持 OpenAI 兼容接口。

### 6.1 核心功能
- **OpenAI 兼容**：通过 `POST /v1/chat/completions` 无缝集成。
- **混合检索**：并行执行向量相似度检索与 Cypher 图谱检索（`mode=hybrid`）。
- **自动化建图**：提供接口将 Markdown 文本自动切分并抽取实体、关系存入 Neo4j。

```bash
cd agent
pip install -r requirements.txt
python server.py
```
默认地址：`http://127.0.0.1:8801`，健康检查：`/api/health`。

### 6.2 主要接口
- `POST /v1/chat/completions`：RAG 问答（OpenAI 格式）。
- `POST /api/query`：自定义检索问答。
- `POST /api/admin/ingest`：文档入库（建图与向量化）。

更多细节请参考 [agent/README.md](./agent/README.md)。
