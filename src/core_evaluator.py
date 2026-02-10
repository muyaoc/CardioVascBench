#!/usr/bin/env python3
"""
核心评估器

- `match_score(pred, gold, icd_name=None, strategy=...)`：
  * strategy="semantic": 返回 (score, tag)，其中 tag∈{"semantic","negative"}
  * strategy="layered" : 返回分层标签 {exact/core/keyword/partial_keyword/semantic_strict/semantic_loose/negative}
- 指标：Top‑k / MRR / weighted_semantic_mean
- 命中口径：semantic 用 tag=="semantic"；layered 用 tag!="negative"
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ------------------------- IO -------------------------

def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ------------------------- 归一化 & 相似度 -------------------------

def _normalize(s: str) -> str:
    s = str(s or "").strip()
    if not s:
        return ""
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("，", ",").replace("；", ";").replace("、", "/")
    s = re.sub(r"[。，、；：！？\.,;:!?]+$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

_ST_MODEL = None
_EMB_CACHE: Dict[str, List[float]] = {}

# New: SapBERT imports and globals
import torch
from transformers import AutoTokenizer, AutoModel

_SAPBERT_TOKENIZER = None
_SAPBERT_MODEL = None
_SAPBERT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _get_sapbert_components():
    global _SAPBERT_TOKENIZER, _SAPBERT_MODEL
    if _SAPBERT_TOKENIZER is not None and _SAPBERT_MODEL is not None:
        return _SAPBERT_TOKENIZER, _SAPBERT_MODEL
    try:
        model_name = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
        _SAPBERT_TOKENIZER = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        _SAPBERT_MODEL = AutoModel.from_pretrained(model_name)
        _SAPBERT_MODEL.to(_SAPBERT_DEVICE)
        _SAPBERT_MODEL.eval()
    except Exception:
        _SAPBERT_TOKENIZER = None
        _SAPBERT_MODEL = None
    return _SAPBERT_TOKENIZER, _SAPBERT_MODEL

def _embed(text: str) -> Optional[List[float]]:
    text = _normalize(text).split("(", 1)[0].strip()
    if not text:
        return [0.0]
    if text in _EMB_CACHE:
        return _EMB_CACHE[text]
    
    tokenizer, model = _get_sapbert_components()
    if tokenizer is None or model is None:
        return None
    
    try:
        with torch.no_grad():
            toks = tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=25,
                return_tensors="pt"
            )
            toks = {k: v.to(_SAPBERT_DEVICE) for k, v in toks.items()}
            outputs = model(**toks)
            cls = outputs.last_hidden_state[:, 0, :]  # [1, H]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            vec = cls.detach().cpu().float().numpy()[0].tolist()
        
        _EMB_CACHE[text] = vec
        return vec
    except Exception:
        return None


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    sa = sum(x * x for x in a)
    sb = sum(x * x for x in b)
    if sa <= 0 or sb <= 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot / math.sqrt(sa * sb)


def _lexical_overlap(a: str, b: str) -> float:
    a = re.sub(r"\W+", "", _normalize(a))
    b = re.sub(r"\W+", "", _normalize(b))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    def ngrams(s: str, n: int = 2) -> List[str]:
        s = s.strip()
        if len(s) < n:
            return [s]
        return [s[i:i+n] for i in range(len(s)-n+1)]
    A = ngrams(a, 2)
    B = ngrams(b, 2)
    inter = 0
    used = [False] * len(B)
    for x in A:
        for j, y in enumerate(B):
            if (not used[j]) and x == y:
                inter += 1
                used[j] = True
                break
    return (2.0 * inter) / (len(A) + len(B)) if (len(A) + len(B)) > 0 else 0.0


def semantic_similarity(a: str, b: Optional[str]) -> float:
    if not b:
        return 0.0
    va = _embed(a)
    vb = _embed(b)
    if va is None or vb is None:
        na = _normalize(a)
        nb = _normalize(b)
        if na == nb or na.split("(", 1)[0] == nb.split("(", 1)[0]:
            return 1.0
        return _lexical_overlap(a, b)
    sim = _cosine(va, vb)
    return max(0.0, min(1.0, sim))  # SapBERT is already in [-1,1] range after normalization, but clamp to [0,1]


# ------------------------- 策略标签 -------------------------
_DEFAULT_LOOSE  = float(os.environ.get("SEM_LOOSE", 0.75))
_DEFAULT_STRICT = float(os.environ.get("SEM_STRICT", 0.90))
_DEFAULT_POS    = float(os.environ.get("SEM_POS", 0.80))

def log_rank_weight(r: int) -> float:
    """Logarithmic discount: 1 / log2(r + 1)."""
    return 1.0 / math.log2(r + 1.0)

def _extract_keywords(text: str) -> set:
    stop_words = {"病", "症", "性", "型", "级", "期", "度", "轻", "中", "重", "急", "慢", "原发", "继发"}
    core = _normalize(text).split("(", 1)[0].strip()
    parts = re.split(r"[\s,\/;，；、]+", core)
    return {w for w in parts if w and len(w) > 1 and w not in stop_words}


def match_score_separate(pred: str, gold: str, icd_name: Optional[str] = None, strategy: str = "semantic") -> Tuple[Tuple[float, str], Tuple[float, str]]:
    """
    返回两个独立的匹配结果：
    - 第一个元组：与原始诊断名称(gold)的匹配结果 (score, tag)
    - 第二个元组：与ICD疾病名称(icd_name)的匹配结果 (score, tag)
    Return separate match results for (Gold, ICD).
    Output format: ((score_gold, tag_gold), (score_icd, tag_icd))
    """
    pn = _normalize(pred)
    gn = _normalize(gold)
    inorm = _normalize(icd_name) if icd_name else ""

    # 计算与原始诊断名称的匹配
    gold_result = _compute_single_match(pred, gold, pn, gn, strategy)
    
    # 计算与ICD疾病名称的匹配
    if icd_name:
        icd_result = _compute_single_match(pred, icd_name, pn, inorm, strategy)
    else:
        icd_result = (0.0, "negative")
    
    return gold_result, icd_result

def _compute_single_match(pred: str, target: str, pred_norm: str, target_norm: str, strategy: str) -> Tuple[float, str]:
    """计算单个目标的匹配分数和标签"""
    if strategy == "semantic":
        s = semantic_similarity(pred, target)
        return (s, "semantic") if s >= _DEFAULT_POS else (s, "negative")

    # Strategy: layered
    # 1. Exact
    if pred_norm and pred_norm == target_norm:
        return 1.0, "exact"
    
    # 2. Core (remove brackets)
    pc = pred_norm.split("(", 1)[0].strip()
    tc = target_norm.split("(", 1)[0].strip()
    if pc and pc == tc:
        return 0.95, "core"

    # 3. Keyword Overlap
    pk = _extract_keywords(pc)
    tk = _extract_keywords(tc)
    overlap = (len(pk & tk) / len(tk)) if tk else 0.0
    if overlap >= 0.6:
        return min(0.95, 0.8 + overlap * 0.15), "keyword"
    elif overlap >= 0.4:
        return 0.6 + overlap * 0.2, "partial_keyword"

    # 4. Semantic fallback
    s = semantic_similarity(pred, target)
    if s >= _DEFAULT_STRICT:
        return s, "semantic_strict"
    elif s >= _DEFAULT_LOOSE:
        return s, "semantic_loose"
    
    return s, "negative"


def match_score(pred: str, gold: str, icd_name: Optional[str] = None, strategy: str = "semantic") -> Tuple[float, str]:
    """
    Combined match score: max of (Gold match, ICD match).
    """
    (sg, tg), (si, ti) = match_score_separate(pred, gold, icd_name, strategy)
    if sg >= si:
        return sg, tg
    return si, ti


# ------------------------- Top‑k / MRR / Soft AUC -------------------------

def _validate_lengths(pred_rankings: List[List[Any]], gold_primary: List[Any]) -> None:
    if len(pred_rankings) != len(gold_primary):
        raise ValueError(
            f"Length mismatch: pred_rankings({len(pred_rankings)}) != gold_primary({len(gold_primary)})"
        )

# # ===== 各种指标计算（底层实现，支持传入不同的 match_fn） =====

def _compute_topk_accuracy_with_match_fn(pred_rankings, gold_primary, icd_names, k: int, match_fn, strategy: str) -> float:
    if not gold_primary or not pred_rankings:
        return 0.0
    hits = 0
    for preds, g, icd in zip(pred_rankings, gold_primary, icd_names):
        ok = False
        for cand in (preds[:k] if preds else []):
            _, tag = match_fn(cand, g, icd)
            if (strategy == "semantic" and tag == "semantic") or (strategy == "layered" and tag != "negative"):
                ok = True
                break
        if ok:
            hits += 1
    return hits / len(gold_primary)

def _compute_mrr_with_match_fn(pred_rankings, gold_primary, icd_names, match_fn, strategy: str) -> float:
    if not gold_primary or not pred_rankings:
        return 0.0
    total = 0.0
    for preds, g, icd in zip(pred_rankings, gold_primary, icd_names):
        rr = 0.0
        for i, cand in enumerate(preds or []):
            _, tag = match_fn(cand, g, icd)
            ok = (tag == "semantic") if strategy == "semantic" else (tag != "negative")
            if ok:
                rr = 1.0 / float(i + 1)
                break
        total += rr
    return total / len(gold_primary)

def _compute_precision_recall_f1_with_match_fn(
    pred_rankings: List[List[str]],
    gold_primary: List[str],
    icd_names: Optional[List[str]],
    match_fn,
    strategy: str = "semantic",
) -> Dict[str, float]:
    tp = fp = fn = 0
    for preds, g, icd in zip(pred_rankings, gold_primary, icd_names):
        is_correct = False
        if preds:
            top1_pred = preds[0]
            _, tag = match_fn(top1_pred, g, icd)
            if (strategy == "semantic" and tag == "semantic") or (strategy == "layered" and tag != "negative"):
                is_correct = True
        
        if is_correct:
            tp += 1
        else:
            fp += 1
            fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "f1_score": f1}

def _avg_weighted_score_with_match_fn(pred_rankings, gold_primary, icd_names, match_fn, rank_weight_fn=log_rank_weight) -> float:
    scores = []
    for preds, g, icd in zip(pred_rankings, gold_primary, icd_names):
        for i, cand in enumerate(preds or []):
            w = rank_weight_fn(i + 1)
            s, _ = match_fn(cand, g, icd)
            scores.append(w * s)
    return (sum(scores) / len(scores)) if scores else 0.0

def _weighted_semantic_sample_score_with_match_fn(preds, gold_name: str, gold_icd_name: Optional[str], match_fn, rank_weight_fn=log_rank_weight) -> float:
    if not preds:
        return 0.0
    best = 0.0
    for i, p in enumerate(preds):
        w = rank_weight_fn(i + 1)
        s, _ = match_fn(p, gold_name, gold_icd_name)
        best = max(best, w * s)
    return best

def _compute_all_metrics_with_match_fn(pred_rankings, gold_primary, icd_names, match_fn, strategy: str) -> Dict[str, Any]:
    """通用的指标计算入口"""
    top1 = _compute_topk_accuracy_with_match_fn(pred_rankings, gold_primary, icd_names, k=1, match_fn=match_fn, strategy=strategy)
    top2 = _compute_topk_accuracy_with_match_fn(pred_rankings, gold_primary, icd_names, k=2, match_fn=match_fn, strategy=strategy)
    top3 = _compute_topk_accuracy_with_match_fn(pred_rankings, gold_primary, icd_names, k=3, match_fn=match_fn, strategy=strategy)
    mrr = _compute_mrr_with_match_fn(pred_rankings, gold_primary, icd_names, match_fn=match_fn, strategy=strategy)

    avg_w_score = _avg_weighted_score_with_match_fn(pred_rankings, gold_primary, icd_names, match_fn=match_fn)
    
    ws_scores = [
        _weighted_semantic_sample_score_with_match_fn(preds, g, icd_names[i] if icd_names else None, match_fn=match_fn)
        for i, (preds, g) in enumerate(zip(pred_rankings, gold_primary))
    ]
    weighted_sem_mean = (sum(ws_scores) / len(ws_scores)) if ws_scores else 0.0

    prf1 = _compute_precision_recall_f1_with_match_fn(pred_rankings, gold_primary, icd_names, match_fn=match_fn, strategy=strategy)

    return {
        "top1_accuracy": top1,
        "top2_accuracy": top2,
        "top3_accuracy": top3,
        "mrr_primary": mrr,
        "precision": prf1["precision"],
        "recall": prf1["recall"],
        "f1_score": prf1["f1_score"],
        "avg_weighted_score": avg_w_score,
        "weighted_semantic_mean": weighted_sem_mean,
    }

# ------------------------- 任务评测 -------------------------

def evaluate_final_diagnosis_separate(pred_rankings, gold_primary, icd_names=None, strategy: str = "semantic") -> Dict[str, Any]:
    """
    评估函数，返回原始诊断名称和ICD疾病名称的分别匹配结果
    """
    _validate_lengths(pred_rankings, gold_primary)
    if icd_names is None:
        icd_names = [""] * len(gold_primary)

    # 计算原始诊断名称的匹配结果
    def _match_gold(p, g, icd):
        res_gold, _ = match_score_separate(p, g, icd, strategy)
        return res_gold
    gold_results = _compute_all_metrics_with_match_fn(pred_rankings, gold_primary, icd_names, _match_gold, strategy)
    
    # 计算ICD疾病名称的匹配结果
    def _match_icd(p, g, icd):
        _, res_icd = match_score_separate(p, g, icd, strategy)
        return res_icd
    icd_results = _compute_all_metrics_with_match_fn(pred_rankings, gold_primary, icd_names, _match_icd, strategy)
    
    # 合并结果
    combined_results = {}
    for key, value in gold_results.items():
        combined_results[f"gold_{key}"] = value
    for key, value in icd_results.items():
        combined_results[f"icd_{key}"] = value
    
    return combined_results

def evaluate_final_diagnosis(pred_rankings, gold_primary, icd_names=None, strategy: str = "semantic") -> Dict[str, Any]:
    """
    评估函数（综合版）：取 Gold 和 ICD 中的最佳匹配。
    """
    _validate_lengths(pred_rankings, gold_primary)
    if icd_names is None:
        icd_names = [""] * len(gold_primary)

    def _match_combined(p, g, icd):
        return match_score(p, g, icd, strategy=strategy)

    return _compute_all_metrics_with_match_fn(pred_rankings, gold_primary, icd_names, _match_combined, strategy)


def summarize_diag_metrics(metrics: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
    # 新版：如果上游提供了多诊断排序的 final_score，优先使用它作为最终分
    if "final_score" in metrics:
        return {"Score_diag": float(metrics.get("final_score", 0.0))}

    c1 = float(weights.get("c1", 0.0))
    c2 = float(weights.get("c2", 0.0))
    c3 = float(weights.get("c3", 0.0))
    
    accuracy = float(metrics.get("top1_accuracy", 0.0))
    mrr = float(metrics.get("mrr_primary", 0.0))
    user_score = float(metrics.get("weighted_semantic_mean", 0.0))
    
    # 使用 accuracy, mrr, user_defined_score
    # 映射关系: c1 -> accuracy, c2 -> mrr, c3 -> user_defined_score
    total_w = c1 + c2 + c3
    score = c1 * accuracy + c2 * mrr + c3 * user_score
    if total_w == 0:
        score = (accuracy + mrr + user_score) / 3.0
    return {"Score_diag": score}


# ------------------------- Ranked multi-diagnosis metrics (nDCG + penalty) -------------------------

def _gt_exp_weight(rank: int, lambda_: float = 0.7) -> float:
    """GT重要性：指数衰减，rank从1开始。"""
    return math.exp(-lambda_ * (rank - 1))


def _pos_discount(pos: int) -> float:
    """预测位置折扣：DCG标准折扣，pos从1开始。"""
    return 1.0 / math.log2(pos + 1.0)


def _best_match_one_pred_to_gt(
    pred: str,
    gt_names: List[str],
    gt_icd_names: Optional[List[str]],
    used_gt_idx: set,
    strategy: str,
) -> Tuple[float, int]:
    """给一个pred找一个尚未被匹配的GT（避免重复计分）。

    返回：(rel, gt_index)。找不到则 (0.0, -1)
    rel 使用 match_score 的 score（0~1）
    """
    best_rel = 0.0
    best_i = -1
    for i, g in enumerate(gt_names):
        if i in used_gt_idx:
            continue
        icd_name = gt_icd_names[i] if (gt_icd_names and i < len(gt_icd_names)) else None
        rel, _ = match_score(pred, g, icd_name, strategy=strategy)
        if rel > best_rel:
            best_rel = rel
            best_i = i
    return best_rel, best_i


def _compute_sample_ndcg_and_penalty(
    preds: List[str],
    gt_names: List[str],
    gt_icd_names: Optional[List[str]] = None,
    K: int = 10,
    lambda_: float = 0.7,
    rel_threshold: float = 0.7,
    beta: float = 0.7,
    strategy: str = "semantic",
) -> Dict[str, float]:
    """
    计算单样本的多诊断排序指标：
    1. nDCG@K: 归一化折损累计增益，衡量排序质量（GT重要性随排名衰减）
    2. PrimaryHit@K: 是否命中主诊断（第一金标准）
    3. Precision@K / Recall@K: 集合检索指标
    4. Final Score: 综合打分 = nDCG * (0.7 + 0.3*PrimaryHit) * (Precision^beta)
    """

    preds = preds or []
    gt_names = gt_names or []
    m = len(gt_names)
    if m == 0:
        return {
            "ndcg@K": 0.0,
            "primary_hit@K": 0.0,
            "precision@K": 0.0,
            "recall@K": 0.0,
            "final_score": 0.0,
        }

    top_preds = preds[:K]

    # -------------------------------------------------------------------------
    # 1. PrimaryHit: 主诊断（GT列表第1个）是否在 Top-K 预测中出现
    # -------------------------------------------------------------------------
    primary_hit = 0.0
    if top_preds:
        for p in top_preds:
            rel0, _ = match_score(p, gt_names[0], gt_icd_names[0] if gt_icd_names else None, strategy=strategy)
            if rel0 >= rel_threshold:
                primary_hit = 1.0
                break

    # -------------------------------------------------------------------------
    # 2. 计算 DCG (Discounted Cumulative Gain)
    #    对每个预测位置 j (1..K)，找一个最佳匹配的 GT (去重)
    #    Gain = rel_score * exp(-lambda * (gt_rank_index))
    #    Discount = 1 / log2(j + 1)
    # -------------------------------------------------------------------------
    dcg = 0.0
    used_gt = set()
    matched_pred = 0
    for j, p in enumerate(top_preds, start=1):
        rel, gi = _best_match_one_pred_to_gt(p, gt_names, gt_icd_names, used_gt, strategy=strategy)
        if gi >= 0:
            used_gt.add(gi)
        # GT重要性随其在金标准列表中的位置衰减 (gi=0是主诊断, 权重最大)
        gain = (rel * _gt_exp_weight(gi + 1, lambda_)) if gi >= 0 else 0.0
        dcg += gain * _pos_discount(j)
        if rel >= rel_threshold and gi >= 0:
            matched_pred += 1

    precision_k = (matched_pred / float(len(top_preds))) if top_preds else 0.0

    # -------------------------------------------------------------------------
    # 3. Recall: 有多少个不同的 GT 被召回 (rel >= threshold)
    # -------------------------------------------------------------------------
    used_gt2 = set()
    recall_hit_gt = set()
    for p in top_preds:
        rel, gi = _best_match_one_pred_to_gt(p, gt_names, gt_icd_names, used_gt2, strategy=strategy)
        if gi >= 0:
            used_gt2.add(gi)
            if rel >= rel_threshold:
                recall_hit_gt.add(gi)
    recall_k = (len(recall_hit_gt) / float(m)) if m > 0 else 0.0

    # -------------------------------------------------------------------------
    # 4. IDCG (Ideal DCG): 理想情况下的最佳排序
    #    将所有 GT 按重要性(权重)从高到低排列，填满前 min(K, m) 个位置
    # -------------------------------------------------------------------------
    idcg = 0.0
    for t in range(1, min(K, m) + 1):
        # 理想情况下，第 t 个位置放第 t 重要的 GT，rel=1.0
        idcg += (1.0 * _gt_exp_weight(t, lambda_)) * _pos_discount(t)

    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    # -------------------------------------------------------------------------
    # 5. Final Score 计算
    #    核心是 nDCG，但对“漏掉主诊断”和“低精确率”进行惩罚
    # -------------------------------------------------------------------------
    final_score = ndcg * (0.7 + 0.3 * primary_hit) * (precision_k ** beta)

    return {
        "ndcg@K": float(ndcg),
        "primary_hit@K": float(primary_hit),
        "precision@K": float(precision_k),
        "recall@K": float(recall_k),
        "final_score": float(final_score),
    }


def evaluate_ranked_diagnoses(
    pred_rankings: List[List[str]],
    gold_ranked_lists: List[List[str]],
    gold_icd_ranked_lists: Optional[List[List[str]]] = None,
    K: int = 10,
    lambda_: float = 0.7,
    rel_threshold: float = 0.7,
    beta: float = 0.7,
    strategy: str = "semantic",
) -> Dict[str, Any]:
    """
    批量评估：多诊断（有序）场景。
    输入：
    - pred_rankings: 模型预测的诊断列表 (每行一个样本)
    - gold_ranked_lists: 金标准诊断列表 (每行一个样本，有序，第一个为主诊断)
    返回：
    - 平均 nDCG@K, PrimaryHit@K, Precision@K, Recall@K, FinalScore
    """
    if len(pred_rankings) != len(gold_ranked_lists):
        raise ValueError(
            f"Length mismatch: pred_rankings({len(pred_rankings)}) != gold_ranked_lists({len(gold_ranked_lists)})"
        )

    ndcgs, phits, precs, recs, finals = [], [], [], [], []
    for idx, preds in enumerate(pred_rankings):
        gt_names = gold_ranked_lists[idx] or []
        gt_icds = gold_icd_ranked_lists[idx] if gold_icd_ranked_lists else None
        s = _compute_sample_ndcg_and_penalty(
            preds=preds,
            gt_names=gt_names,
            gt_icd_names=gt_icds,
            K=K,
            lambda_=lambda_,
            rel_threshold=rel_threshold,
            beta=beta,
            strategy=strategy,
        )
        ndcgs.append(s["ndcg@K"])
        phits.append(s["primary_hit@K"])
        precs.append(s["precision@K"])
        recs.append(s["recall@K"])
        finals.append(s["final_score"])

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    return {
        "K": K,
        "lambda": lambda_,
        "rel_threshold": rel_threshold,
        "beta": beta,
        "ndcg@K": _mean(ndcgs),
        "primary_hit@K": _mean(phits),
        "precision@K": _mean(precs),
        "recall@K": _mean(recs),
        "final_score": _mean(finals),
    }

