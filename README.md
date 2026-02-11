# CardioVascBench 心脑血管疾病大模型评测基准

面向心脑血管疾病场景的中文评测基准。依据真实心内科数据（如 `data/source_data/source_data.json`）构建评估集，输入病历文书与心电检查相关信息，评估待测大语言模型的“最终诊断”能力与一致性。

**数据下载 (Data Download)**：[CardioVascBench 数据集 (OpenI 启智社区)](https://openi.pcl.ac.cn/explore/datasets/cindyc/CardioVascBench)

本 README 面向**内部使用者与开发者**，重点说明：

- 评测流程与数据流向；
- 诊断匹配与归一化算法细节；
- 各类 AUC / Top‑k / MRR 等指标的严格数学定义；
- 综合得分 `Score_diag` / `Score_total` 的计算规则与配置方法。

---

## 1. 目录结构与角色说明

```text
Code/
├── config/
│   ├── eval_llm_api.yml          # 待测 LLM 评估的 API 配置
│   ├── paths.yml                 # 所有路径信息
│   └── weights.yml               # 指标权重信息
├── data/
│   ├── source_data/              # 真实心内科数据与转换脚本
│   ├── llm_eval_data/            # 基准评估数据（eval/test）
│   └── llms_answers_data_generator.py # 生成模型答案的脚本
├── src/
│   ├── core_evaluator.py         # 核心评测算法（归一化 / 匹配 / 各类指标）
│   └── evaluate_llm.py           # 评估待测 LLM（读取配置，脚本调度）
├── agent/                        # [新增] 图谱 RAG Agent 模块 (FastAPI 服务)
├── requirements.txt              # 项目依赖
└── results/
    └── llms_evaluation_results/  # [输出] LLM 评测结果 (JSON / PNG / CSV)
    
```

---

## 2. 环境准备与依赖

### 2.1 Python 与依赖

- 推荐 Python 版本：`>= 3.9`；
- 安装依赖：

```bash
pip install -r requirements.txt
```

若使用默认的语义相似度模型（SapBERT），需要较新的 `torch` 与 `transformers`，以及可用的 GPU（否则自动回退为 CPU / 退化字符串相似度）。

### 2.2 配置文件总览

所有路径、权重与模型相关参数**必须**从 `config/` 读取，不允许在代码中硬编码：

- `config/paths.yml`：数据 / 答案 / 结果路径；
- `config/weights.yml`：各指标权重与总分系数；
- `config/eval_llm_api.yml`：API 或模型名称等元信息（评估阶段只用于记录）。

示例（简化）：

```yaml
# config/paths.yml
llm_eval4_data_paths:
  final_diagnosis_eval: data/llm_eval_data/eval_data/private_diagnosis_eval.json

llms_answers_data_base_dir: data/llms_answers_data

output_paths:
  llms_evaluation_results: results/llms_evaluation_results
```

```yaml
# config/weights.yml
gamma: 1.0   # 总分缩放因子：Score_total = gamma * Score_diag

# 诊断任务综合分：Score_diag = c1 * Top3 + c2 * AUC + c3 * MRR
c1: 0.00     # Accuracy (Top-1) 权重
c2: 1.00     # F1-score 权重
c3: 0.00     # User Defined Score (weighted_semantic_mean) 权重
```

`config/eval_llm_api.yml` 主要用于记录当前评测使用的模型名等元信息（评估时不会主动调 API）：

```yaml
api_url: https://api.example.com/v1
model_name: your-model-name
api_key: YOUR_API_KEY   # 可留空，评估阶段不会访问
```

---

## 3. 评测流程（数据流与运行方式）

整体流程分为三步：

1. **构建评估集**：由 `data/source_data/` 中的真实病历 / 检查数据，经清洗脚本生成 `llm_eval_data/eval_data/*.json`；
2. **生成模型答案**：调用各模型，对评估样本生成“最终诊断”候选列表，保存到 `data/llms_answers_data/<model_name>/final_diagnosis_pred.json`；
3. **离线评估**：运行 `src/evaluate_llm.py`，读取金标准与模型答案，调用 `core_evaluator.py` 计算各类指标与总分，并输出 JSON / 图表 / 调试 CSV。

### 3.1 快速开始（已有人生成好答案的场景）

```bash
cd ECGbenchmark

# 如需先生成某个模型的回答（示意）
python data/llms_answers_data_generator.py

# 对单个模型进行评估（默认 strategy=semantic, auc_alias=pr）
python src/evaluate_llm.py --models your-model-name

# 或一次评估多个模型（英文逗号分隔）
python src/evaluate_llm.py --models Apollo-72B,qwen3
```

脚本会从：

- `paths.yml` 中读取：
  - `llm_eval4_data_paths.final_diagnosis_eval`：评估用 JSON（金标准）；
  - `llms_answers_data_base_dir`：模型答案根目录；
  - `output_paths.llms_evaluation_results`：结果输出目录；
- `weights.yml` 中读取：
  - `gamma, c1, c2, c3`：综合分加权与缩放；
- `eval_llm_api.yml` 中读取：
  - `model_name, api_url` 等用于写入结果 JSON 的元信息。

输出结果主要包括：

- `results/llms_evaluation_results/llm_evaluation_metrics_<model>-<strategy>.json`
- `results/llms_evaluation_results/figures_<model>-<strategy>/*.png`
- `results/llms_evaluation_results/debug_evaluation_<model>-<strategy>_*.csv`

---

## 4. 匹配与归一化算法（core_evaluator）

### 4.1 文本归一化 `_normalize`

函数 `_normalize(s: str)` 负责对诊断文本做轻量归一化：

- 去除首尾空白；
- 中/英文括号、逗号、分号的统一（如 `（`→`(`，`，`→`,`）；
- 去除末尾句号 / 感叹号等标点；
- 合并连续空白。

归一化后字符串再进入后续匹配与相似度计算流程。

### 4.2 语义相似度：SapBERT + 字符级回退

核心函数 `semantic_similarity(a, b)` 的逻辑：

1. 对 `a, b` 各自取归一化后去括号前的“主干部分”作为文本；
2. 调用 `_embed(text)` 使用 SapBERT 模型：
   - 模型：`cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR`；
   - 使用 `[CLS]` 向量并做 L2 归一化；
   - 使用余弦相似度 `_cosine(a, b)` 得分；
3. 若嵌入加载失败（如无 GPU / 网络问题），退化为：
   - 若归一化字符串相等或去括号主干相等：相似度设为 1.0；
   - 否则使用 `_lexical_overlap`（基于 2‑gram 的字符重叠度）；
4. 最终相似度被裁剪在 `[0, 1]` 区间。

### 4.3 语义 / 分层匹配标签 `match_score`

核心接口（简化签名）：

```python
match_score(pred: str, gold: str, icd_name: Optional[str] = None, strategy: str = "semantic") -> Tuple[float, str]
```

#### 4.3.1 semantic 策略

- 计算：

  - `s_gold  = semantic_similarity(pred, gold)`
  - `s_icd   = semantic_similarity(pred, icd_name)`（若存在 ICD 名称）
  - `s       = max(s_gold, s_icd)`

- 若 `s >= SEM_POS`（默认 0.80），返回 `(s, "semantic")`；否则 `(s, "negative")`。
- 仅提供“是否语义命中”的二值标签。

#### 4.3.2 layered 策略

分层标签细化了“近似命中”的不同强度：

1. exact：归一化后的 `pred` 与 `gold`（或 ICD 名）完全一致，得分 1.0；
2. core：去括号主干（如“心肌梗死(急性期)”→“心肌梗死”）一致，得分 0.95；
3. keyword / partial_keyword：
   - 使用 `_extract_keywords` 抽取诊断中的关键字（排除“病 / 症 / 性 / 级 / 期 / 度 / 急 / 慢”等弱词）；
   - 计算关键词重叠率，若与 gold / ICD 的关键字重叠率：
     - ≥ 0.6：标签 `"keyword"`，得分约 0.8~0.95；
     - ∈ [0.4, 0.6)：标签 `"partial_keyword"`，得分约 0.68~0.8；
4. 语义两档：
   - 计算语义相似度 `s`；
   - 若 `s >= SEM_STRICT`（默认 0.90）：`"semantic_strict"`；
   - 若 `s >= SEM_LOOSE`（默认 0.75）：`"semantic_loose"`；
   - 否则 `"negative"`。

#### 4.3.3 命中口径与环境变量

- `strategy="semantic"`：标签为 `"semantic"` 视为命中；
- `strategy="layered"`：标签不为 `"negative"` 视为命中（含 exact / core / keyword / semantic_* 等）。

可通过环境变量调整关键阈值（未设置则使用默认值）：

- `SEM_POS`（默认 0.80）：semantic 策略下的正样本阈值；
- `SEM_STRICT`（默认 0.90）：layered 策略中“严格语义命中”的阈值；
- `SEM_LOOSE`（默认 0.75）：layered 策略中“宽松语义命中”的阈值。

### 4.4 排名权重 `log_rank_weight`

使用对数折扣函数 `log_rank_weight(r)`，模拟用户浏览行为的注意力衰减：

```python
weight = 1.0 / log2(rank + 1)
```

- 第 1 名：1.00
- 第 2 名：0.63
- 第 3 名：0.50
- 第 4 名：0.43
- ...

该权重主要用于计算 `weighted_semantic_mean` 和调试 CSV 中的 `scores_per_pred`。

---

## 5. 指标定义 (Ranked Multi-Diagnosis & Best-Match)

### 5.1 多诊断排序指标 (Ranked Multi-Diagnosis)

当一个病例存在多个正确诊断（如：房颤、心衰、高血压）时，我们使用信息检索领域的标准指标来评估模型生成的诊断列表。

#### 5.1.1 nDCG@K (归一化折损累计增益)
衡量排序质量。不仅要求模型预测正确，还要求将正确的诊断排在前面。

- **Gain (增益)**: `rel_score * _gt_exp_weight(gt_rank)`
  - `rel_score`: 预测值与金标准的语义相似度（0~1）。
  - `_gt_exp_weight(rank)`: GT 重要性权重，采用指数衰减 `exp(-lambda * (rank - 1))`，默认 `lambda=0.7`。其中 rank 为金标准列表中的排名（从1开始），越靠前的主诊断权重越大。
- **Discount (折损)**: `_pos_discount(pred_pos)`
  - `_pos_discount(pos)`: 预测位置折扣，采用标准 DCG 折扣 `1 / log2(pos + 1.0)`。模型把正确答案排得越靠后，该分值越低。

#### 5.1.2 PrimaryHit@K
衡量模型是否命中了“第一主诊断”。
- 若 Top-K 预测中，有任一候选与金标准的第一项相似度 > `rel_threshold`，则记为 1.0，否则 0.0。

#### 5.1.3 Precision@K & Recall@K
- **Precision@K**: 前 K 个预测中有多少是正确的。
- **Recall@K**: 金标准列表中的诊断有多少被召回了。

#### 5.1.4 Final Score (综合分)
结合排序质量、主诊断命中率和精确率的综合打分：

```text
FinalScore = nDCG * (0.7 + 0.3 * PrimaryHit) * (Precision ^ beta)
```

### 5.2 单诊断最佳匹配指标 (Best-Match / Combined)

仅关注模型是否“沾边”金标准中的任何一个诊断（通常用于兼容旧指标）。

- **Top-k Accuracy**: 前 k 个预测中是否包含正确答案。
- **MRR (Mean Reciprocal Rank)**: 第一个正确答案排名的倒数。
- **Weighted Semantic Mean**: 所有预测的加权语义相似度均值。

---

## 6. 综合评分：Score_diag 与 Score_total

### 6.1 Score_diag（单任务诊断综合分）

计算函数：`summarize_diag_metrics`。

**优先逻辑**:
若启用了多诊断排序评估，则直接使用其 `FinalScore`：
```python
Score_diag = metrics["final_score"]
```

若未启用（回退模式），则使用加权公式：
```python
Score_diag = c1 * Accuracy + c2 * MRR + c3 * UserScore
```
*(权重 c1/c2/c3 在 config/weights.yml 中定义)*

### 6.2 Score_total（总分）

从 `weights.yml` 读取缩放因子 `gamma`，总分计算为：

```text
Score_total = gamma * Score_diag
```

目前仅包含“最终诊断”一个子任务，因此：

- `summary_scores.Score_diag`：该任务综合分；
- `Score_total`：当前任务下的总体分数（便于未来扩展多任务）。

---

## 7. 结果文件结构说明

### 7.1 JSON：`llm_evaluation_metrics_<model>-<strategy>.json`

典型结构：

```json
{
  "model_name": "your-model-name",
  "summary_scores": {
    "Score_total": 0.84
  },
  "ranked_multi_diag": {
    "ndcg@K": 0.75,
    "primary_hit@K": 1.0,
    "precision@K": 0.60,
    "recall@K": 0.80,
    "final_score": 0.84
  },
  "combined": {
    "top1_accuracy": 0.72,
    "top2_accuracy": 0.84,
    "mrr_primary": 0.65,
    "weighted_semantic_mean": 0.78
  },
  "gold": { "...": "..." },
  "icd": { "...": "..." }
}
```

说明：

- `ranked_multi_diag`：多诊断排序评估结果（主要参考）。
- `combined`：兼容旧版指标（Top-1 Accuracy 等），仅供参考。
- `desc_metrics_separate`（拆分为 `gold` 和 `icd`）：辅助分析。

### 7.2 图表：`figures_<model>-<strategy>/*.png`

- `topk_accuracy_bar_*.png`：Top‑k 准确率柱状图；

### 7.3 调试 CSV：`debug_evaluation_<model>-<strategy>_*.csv`

由 `generate_all_debug_csvs` 生成三种版本：

- `<strategy>.csv`：匹配策略为“max（gold vs ICD 取分数更高者）”；
- `<strategy>_gold.csv`：仅基于原始诊断名匹配；
- `<strategy>_icd.csv`：仅基于 ICD 疾病名匹配。

每个 CSV 中包含：

- 每条样本的 `id`、金标准诊断名、ICD 名；
- 每个候选的语义相似度 `similarities_per_pred`；
- 加权分 `scores_per_pred`（即 `w(r) * s_semantic`）；
- Top‑1 / 最佳匹配的得分与标签类型（`exact/core/semantic_*` 等）；
- 是否存在 exact/core 级别命中。

---

## 8. 常见问题与最佳实践

- 不硬编码配置：所有路径、权重、模型名等全部由 `config/*.yml` 指定；
- 依赖安装问题：优先排查 Python 版本与虚拟环境，必要时单独测试 `torch` 和 `transformers` 加载；
- 路径问题：优先通过修改 `config/paths.yml` 适配本地目录，而非改动代码；
- 评测复现性：
  - 固定 `weights.yml` 与评估数据版本；
---

## 8. Medical Graph RAG Agent (agent/)

该模块提供基于 **Neo4j + 向量检索** 的医学知识图谱 RAG 能力，支持 OpenAI 兼容接口。

### 8.1 核心功能
- **OpenAI 兼容**：通过 `POST /v1/chat/completions` 无缝集成。
- **混合检索**：并行执行向量相似度检索与 Cypher 图谱检索（`mode=hybrid`）。
- **自动化建图**：提供接口将 Markdown 文本自动切分并抽取实体、关系存入 Neo4j。

### 8.2 快速运行
```bash
cd agent
pip install -r requirements.txt
python server.py
```
默认地址：`http://127.0.0.1:8801`，健康检查：`/api/health`。

### 8.3 主要接口
- `POST /v1/chat/completions`：RAG 问答（OpenAI 格式）。
- `POST /api/query`：自定义检索问答。
- `POST /api/admin/ingest`：文档入库（建图与向量化）。

更多细节请参考 [agent/README.md](file:///e:/mychen/CVDBenchmark-OpenSource/Code/agent/README.md)。
