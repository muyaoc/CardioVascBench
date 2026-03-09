# CardioVascBench 心脑血管疾病大模型评测基准

[English Version (README_EN.md)](./README_EN.md)

面向心脑血管疾病场景的中文评测基准。依据真实心内科数据构建评估集，评估待测大语言模型的“最终诊断”能力与一致性。

**数据下载 (Data Download)**：[CardioVascBench 数据集 (OpenI 启智社区)](https://openi.pcl.ac.cn/datasets/detail/cindyc/CardioVascBench)

本 README 旨在为**用户（尤其是临床医生）**提供快速了解和操作本评测基准的指南。

---

## 1. 目录结构与角色说明

```text
CARDIOVASCBENCH/
├── config/
│   ├── eval_llm_api.yml          # 评估模型参数
│   ├── paths.yml                 # 所有路径信息
│   └── weights.yml               # 指标权重信息
├── data/
│   ├── source_data/              # 真实心内科数据与转换脚本
│   ├── llm_eval_data/            # 基准评估数据（eval/test）
│   └── llms_answers_data_generator.py # 生成模型答案的脚本
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

- `torch` 可能需要根据你的运行环境选择安装方式；如果使用 GPU，建议按 PyTorch 官方说明选择匹配的 CUDA 版本安装。
- `src/core_evaluator.py` 默认使用 SapBERT 模型 `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR` 做语义匹配。首次运行时，`transformers` 会尝试联网从 Hugging Face 下载模型文件，因此需要可访问外网。
- SapBERT 模型地址：[`cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR`](https://huggingface.co/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR)
- 如果当前环境无法联网，可以先在有网络的机器手动下载该模型，再拷贝到本地服务器；随后将 `src/core_evaluator.py` 中的模型名替换为本地目录，或提前放入 Hugging Face 缓存目录后再运行。

### 2.2 配置文件总览

所有路径、权重与模型相关参数**必须**从 `config/` 读取，不允许在代码中硬编码：

- `config/paths.yml`：数据 / 答案 / 结果路径；
- `config/weights.yml`：各指标权重与总分系数；

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

### 3.2 生成模型答案

当前版本的 `data/llms_answers_data_generator.py` 默认直接读取：

- `config/eval_llm_api.yml`：API 地址、密钥、模型名与全部请求参数；
- `config/paths.yml`：评测输入路径与答案输出目录。

最简单的运行方式是：

```bash
python data/llms_answers_data_generator.py
```

如需显式指定配置文件，也可以运行：

```bash
python data/llms_answers_data_generator.py \
  --model-cfg config/eval_llm_api.yml \
  --paths-cfg config/paths.yml
```

运行前请先修改 `config/eval_llm_api.yml`。该文件至少需要配置：

1.  **API 信息**
    - `api_url`：OpenAI 兼容服务地址。可填写基础地址，也可填写完整接口地址；脚本会自动归一化。
    - `api_key`：访问密钥。

2.  **模型与请求参数**
    - `model_name`：模型名称。
    - `endpoint`：`chat` 或 `completions`。
    - `use_client_templates`：是否在本地把 `messages` 渲染成 `prompt`。
    - `supports_system_role`：服务端是否支持 `system` 角色。
    - `json_mode`、`stops`、`max_tokens`、`token_param`、`temperature`、`timeout_seconds`、`max_retries`、`backoff_seconds`、`template`。

3.  **路径配置**
    - `config/paths.yml` 中的 `llm_eval4_data_paths.final_diagnosis_eval`：评测输入文件。
    - `config/paths.yml` 中的 `llms_answers_data_base_dir`：答案输出根目录。

脚本会先做一次 `/v1/models` 连通性检查；通过后逐条生成结果，并输出到：

```text
data/llms_answers_data/<model_name>/final_diagnosis_pred.json
```

同时会在同目录记录错误日志 `api_errors.log`。

> 注意：`paths.yml` 中若使用相对路径，请在项目根目录执行命令。

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
