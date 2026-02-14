# CardioVascBench Cardiovascular Disease Large Model Evaluation Benchmark

A Chinese evaluation benchmark oriented towards cardiovascular disease scenarios. Built on real cardiology data (e.g., `data/source_data/source_data.json`), it evaluates the "final diagnosis" capability and consistency of Large Language Models (LLMs) by inputting medical records and ECG-related information.

**Data Download**: [CardioVascBench Dataset (OpenI)](https://openi.pcl.ac.cn/explore/datasets/cindyc/CardioVascBench)

This README is intended for **internal users and developers**, focusing on:

- Evaluation workflow and data flow;
- Diagnosis matching and normalization algorithm details;
- Strict mathematical definitions of various metrics (AUC / Top-k / MRR, etc.);
- Calculation rules and configuration methods for composite scores `Score_diag` / `Score_total`.

---

## 1. Directory Structure and Roles

```text
Code/
├── config/
│   ├── eval_llm_api.yml          # API config for the LLM under evaluation
│   ├── paths.yml                 # All path information
│   └── weights.yml               # Metric weight information
├── data/
│   ├── source_data/              # Real cardiology data and conversion scripts
│   ├── llm_eval_data/            # Benchmark evaluation data (eval/test)
│   └── llms_answers_data_generator.py # Script to generate model answers
├── src/
│   ├── core_evaluator.py         # Core evaluation algorithms (normalization / matching / metrics)
│   └── evaluate_llm.py           # Evaluate the LLM (reads config, invokes core_evaluator)
├── agent/                        # [New] Medical Graph RAG Agent module (FastAPI service)
├── requirements.txt              # Project dependencies
└── results/
    └── llms_evaluation_results/  # [Output] LLM evaluation results (JSON / PNG / CSV)
```

---

## 2. Environment Preparation and Dependencies

### 2.1 Python and Dependencies

- Recommended Python Version: `>= 3.9`;
- Install Dependencies:

```bash
pip install -r requirements.txt
```

If using the default semantic similarity model (SapBERT), newer versions of `torch` and `transformers` are required, along with an available GPU (otherwise it automatically falls back to CPU / character-level similarity).

### 2.2 Configuration Overview

All paths, weights, and model-related parameters **must** be read from `config/`; hardcoding in the code is not allowed:

- `config/paths.yml`: Paths for data / answers / results;
- `config/weights.yml`: Weights for each metric and total score coefficients;
- `config/eval_llm_api.yml`: Meta-information like API URL or model name (used for recording during evaluation).

Example (simplified):

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
gamma: 1.0   # Total score scaling factor: Score_total = gamma * Score_diag

# Diagnostic task composite score: Score_diag = c1 * Top3 + c2 * AUC + c3 * MRR
c1: 0.00     # Accuracy (Top-1) weight
c2: 1.00     # F1-score weight
c3: 0.00     # User-defined score (weighted_semantic_mean) weight
```

---

## 3. Evaluation Workflow (Data Flow and Execution)

The overall process is divided into three steps:

1. **Build Evaluation Set**: Generated from real medical records / ECG data in `data/source_data/` via cleaning scripts into `llm_eval_data/eval_data/*.json`;
2. **Generate Model Answers**: Call various models to generate "final diagnosis" candidate lists for evaluation samples, saved to `data/llms_answers_data/<model_name>/final_diagnosis_pred.json`;
3. **Offline Evaluation**: Run `src/evaluate_llm.py`, which reads the gold standard and model answers, calls `core_evaluator.py` to calculate various metrics and total scores, and outputs JSON / charts / debug CSVs.

### 3.1 Quick Start (Scenarios with pre-generated answers)

```bash
cd Code

# Generate answers for a model (example)
# Defaults to --model-cfg config/eval_llm_api.yml --paths-cfg config/paths.yml
python data/llms_answers_data_generator.py

# Evaluate a single model (default strategy=semantic, auc_alias=pr)
python src/evaluate_llm.py --models your-model-name

# Or evaluate multiple models at once (comma-separated)
python src/evaluate_llm.py --models Apollo-72B,qwen3
```

The script will:

- Read from `paths.yml`:
  - `llm_eval4_data_paths.final_diagnosis_eval`: Evaluation JSON (Gold Standard);
  - `llms_answers_data_base_dir`: Base directory for model answers;
  - `output_paths.llms_evaluation_results`: Output directory for results;
- Read from `weights.yml`:
  - `gamma, c1, c2, c3`: Composite score weighting and scaling;
- Read from `eval_llm_api.yml`:
  - `model_name, api_url`, etc., for writing into the result JSON.

Output results mainly include:

- `results/llms_evaluation_results/llm_evaluation_metrics_<model>-<strategy>.json`
- `results/llms_evaluation_results/figures_<model>-<strategy>/*.png`
- `results/llms_evaluation_results/debug_evaluation_<model>-<strategy>_*.csv`

---

## 4. Matching and Normalization Algorithm (core_evaluator)

### 4.1 Text Normalization `_normalize`

The `_normalize(s: str)` function is responsible for lightweight normalization of diagnostic text:

- Trimming leading/trailing whitespace;
- Unifying Chinese/English parentheses, commas, and semicolons (e.g., `（`→`(`);
- Removing trailing punctuation like periods or exclamation marks;
- Merging consecutive whitespaces.

The normalized string then enters the subsequent matching and similarity calculation process.

### 4.2 Semantic Similarity: SapBERT + Character-level Fallback

Logic of the core function `semantic_similarity(a, b)`:

1. Take the "core part" (before parentheses) of the normalized `a` and `b` as the text;
2. Call `_embed(text)` using the SapBERT model:
   - Model: `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR`;
   - Uses `[CLS]` vector and performs L2 normalization;
   - Uses cosine similarity `_cosine(a, b)` for the score;
3. If embedding fails to load (e.g., no GPU / network issues), it falls back to:
   - If normalized strings match or core parts match: similarity set to 1.0;
   - Otherwise, use `_lexical_overlap` (character overlap based on 2-grams);
4. Final similarity is clipped to the `[0, 1]` range.

### 4.3 Semantic / Layered Matching Labels `match_score`

Core interface (simplified signature):

```python
match_score(pred: str, gold: str, icd_name: Optional[str] = None, strategy: str = "semantic") -> Tuple[float, str]
```

#### 4.3.1 semantic Strategy

- Calculation:
  - `s_gold = semantic_similarity(pred, gold)`
  - `s_icd = semantic_similarity(pred, icd_name)` (if ICD name exists)
  - `s = max(s_gold, s_icd)`
- If `s >= SEM_POS` (default 0.80), returns `(s, "semantic")`; otherwise `(s, "negative")`.
- Only provides a binary label: "semantic hit" or not.

#### 4.3.2 layered Strategy

Layered labels refine the intensity of "approximate hits":

1. **exact**: Normalized `pred` matches `gold` (or ICD name) exactly, score 1.0;
2. **core**: Core parts (before parentheses) match (e.g., "Myocardial Infarction(Acute)" → "Myocardial Infarction"), score 0.95;
3. **keyword / partial_keyword**:
   - Uses `_extract_keywords` to extract keywords from the diagnosis (excluding weak words like "disease", "type", "acute", etc.);
   - Calculates keyword overlap rate with gold / ICD names:
     - ≥ 0.6: Label `"keyword"`, score approx. 0.8~0.95;
     - ∈ [0.4, 0.6): Label `"partial_keyword"`, score approx. 0.68~0.8;
4. **Semantic Levels**:
   - Calculates semantic similarity `s`;
   - If `s >= SEM_STRICT` (default 0.90): `"semantic_strict"`;
   - If `s >= SEM_LOOSE` (default 0.75): `"semantic_loose"`;
   - Otherwise `"negative"`.

---

## 5. Metric Definitions (Ranked Multi-Diagnosis & Best-Match)

### 5.1 Ranked Multi-Diagnosis Metrics

When a case has multiple correct diagnoses (e.g., Atrial Fibrillation, Heart Failure, Hypertension), we use standard Information Retrieval metrics.

#### 5.1.1 nDCG@K (Normalized Discounted Cumulative Gain)
Measures ranking quality. Requires not just correct predictions, but also that correct diagnoses are ranked higher.

- **Gain**: `rel_score * _gt_exp_weight(gt_rank)`
  - `rel_score`: Semantic similarity between prediction and gold standard (0~1).
  - `_gt_exp_weight(rank)`: GT importance weight using exponential decay `exp(-lambda * (rank - 1))`, default `lambda=0.7`. Rank starts from 1 (the primary diagnosis has the highest weight).
- **Discount**: `_pos_discount(pred_pos)`
  - `_pos_discount(pos)`: Predicted position discount using standard DCG discount `1 / log2(pos + 1.0)`. Correct answers ranked lower receive a lower discount score.

#### 5.1.2 PrimaryHit@K
Measures whether the model hit the "Primary Diagnosis" (the first item in the gold standard list).

#### 5.1.3 Precision@K & Recall@K
- **Precision@K**: How many of the top K predictions are correct.
- **Recall@K**: How many of the gold standard diagnoses were retrieved.

#### 5.1.4 Final Score
A comprehensive score combining ranking quality, primary diagnosis hit rate, and precision:

```text
FinalScore = nDCG * (0.7 + 0.3 * PrimaryHit) * (Precision ^ beta)
```

---

## 6. Composite Scoring: Score_diag and Score_total

### 6.1 Score_diag (Diagnostic Task Composite Score)

Calculation function: `summarize_diag_metrics`.

**Priority Logic**:
If Ranked Multi-Diagnosis `final_score` is provided, use it directly:
```python
Score_diag = metrics["final_score"]
```

### 6.2 Score_total (Total Score)

Calculated from `weights.yml` using the scaling factor `gamma`:
```text
Score_total = gamma * Score_diag
```

---

## 7. Result File Structure

### 7.1 JSON: `llm_evaluation_metrics_<model>-<strategy>.json`
Detailed metrics including `ranked_multi_diag` (preferred reference) and `combined` (legacy compatibility).

### 7.2 Debug CSV: `debug_evaluation_<model>-<strategy>_*.csv`
Generated in three versions:
- `<strategy>.csv`: Max of Gold vs. ICD matching;
- `<strategy>_gold.csv`: Matching against original diagnosis names only;
- `<strategy>_icd.csv`: Matching against ICD names only.

---

## Medical Graph RAG Agent (agent/)

This module provides Medical Knowledge Graph RAG capabilities based on **Neo4j + Vector Retrieval**, supporting an OpenAI-compatible interface.

### 8.1 Core Features
- **OpenAI Compatible**: Seamless integration via `POST /v1/chat/completions`.
- **Hybrid Retrieval**: Parallel execution of vector similarity retrieval and Cypher graph retrieval (`mode=hybrid`).
- **Automated Graph Building**: Interfaces to automatically split Markdown text and extract entities/relationships into Neo4j.

### 8.2 Quick Start
```bash
cd agent
pip install -r requirements.txt
python server.py
```
Default URL: `http://127.0.0.1:8801`, health check: `/api/health`.

### 8.3 Main Endpoints
- `POST /v1/chat/completions`: RAG Q&A (OpenAI format).
- `POST /api/query`: Custom retrieval Q&A.
- `POST /api/admin/ingest`: Document ingestion (graph building and vectorization).

For more details, please refer to [agent/README.md](file:///e:/mychen/CVDBenchmark-OpenSource/Code/agent/README.md).
