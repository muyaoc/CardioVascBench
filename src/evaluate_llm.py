"""
大模型评估模块

- 仅加载最终诊断评估集与模型答案，计算并保存相关指标与总分。
- 依赖 core_evaluator 的统一接口：evaluate_final_diagnosis / match_score。
- 可视化：Top‑k Accuracy 柱状图。
- 不对诊断名做 normalize，直接原样传给 core_evaluator（其内部自带归一化）。
- 新增参数：--strategy {semantic,layered}（默认 semantic）。所有输出文件名追加 "-<strategy>" 后缀。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from typing import Any, Dict, List, Tuple

import yaml

from core_evaluator import (
    evaluate_final_diagnosis,
    evaluate_final_diagnosis_separate,
    evaluate_ranked_diagnoses,
    summarize_diag_metrics,
    save_json,
    match_score,
    match_score_separate,
    log_rank_weight,
)


# 轻量可视化依赖（若环境缺失则跳过绘图）
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_fig(fig_dir: str, filename: str) -> str:
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, filename)
    try:
        plt.tight_layout()
        plt.savefig(path, dpi=150)
    except Exception:
        pass
    finally:
        try:
            plt.close()
        except Exception:
            pass
    return path


def plot_topk_accuracy(diag_metrics: Dict[str, Any], fig_dir: str, suffix: str) -> None:
    if plt is None:
        return
    vals = [
        float(diag_metrics.get("top1_accuracy", 0.0)),
        float(diag_metrics.get("top2_accuracy", 0.0)),
        float(diag_metrics.get("top3_accuracy", 0.0)),
    ]
    labels = ["Top-1", "Top-2", "Top-3"]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, vals)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Top-k Accuracy")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, min(0.98, v + 0.02), f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    _save_fig(fig_dir, f"topk_accuracy_bar_{suffix}.png")


# ===== CSV 调试 =====
def generate_debug_csv_with_match_type(
    out_dir: str,
    model_name: str,
    gold_ids: List[str],
    diagnosis_gold_ranked_lists: List[List[str]],
    diagnosis_icd_ranked_lists: List[List[str]],
    diagnosis_pred_rankings: List[List[str]],
    eval_data: List[Dict[str, Any]],
    match_type: str = "max"  # "max", "gold", "icd"
) -> None:
    """生成CSV调试文件：逐条给出每个候选的相似度(s)与加权分(rank_weight*s)。"""
    # Create debug subdirectory for this model
    debug_dir = os.path.join(out_dir, "debug", model_name.split("-")[0])  # Remove suffix like "-semantic"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Add match type suffix to filename
    suffix = f"_{match_type}" if match_type != "max" else ""
    csv_path = os.path.join(debug_dir, f"debug_evaluation_{model_name}{suffix}.csv")

    # id -> eval_data
    eval_data_by_id: Dict[str, Dict[str, Any]] = {}
    for item in eval_data:
        eval_data_by_id[str(item.get("id", ""))] = item

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = [
            'id',
            'eval_diagnosis_name',
            'icd_disease_name',
            'pred_top1',
            'pred_top2',
            'pred_top3',
            'pred_full_rankings',
            # NEW: 逐候选的 semantic 相似度（match_score 的 s）
            'similarities_per_pred',
            # （可选）逐候选的加权分 rank_weight*similarity，若不想要可删除此列
            'scores_per_pred',
            'similarity_score_top1',
            'top1_match_type',
            'similarity_score_max',
            'best_match_pred',
            'best_match_rank',
            'best_match_type',
            'exact_match_found'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, (sample_id, gold_list, icd_list, pred_rankings) in enumerate(
            zip(gold_ids, diagnosis_gold_ranked_lists, diagnosis_icd_ranked_lists, diagnosis_pred_rankings)
        ):
            # Display strings (joined by |)
            gold_display = " | ".join(gold_list)
            icd_display = " | ".join(icd_list)

            # 预测结果
            pred_top1 = pred_rankings[0] if pred_rankings else ""
            pred_top2 = pred_rankings[1] if len(pred_rankings) > 1 else ""
            pred_top3 = pred_rankings[2] if len(pred_rankings) > 2 else ""
            pred_full = " | ".join(pred_rankings) if pred_rankings else ""

            # 计算逐候选的相似度与加权分
            sims: List[str] = []
            weighted_scores: List[str] = []
            best_s, best_pred, best_rank, best_tag = 0.0, "", 0, "none"
            exact_found = False
            
            # Define match function based on match_type
            def get_match_score(cand: str, gold: str, icd: str, strategy: str) -> Tuple[float, str]:
                if match_type == "max":
                    return match_score(cand, gold, icd, strategy=strategy)
                elif match_type == "gold":
                    gold_result, _ = match_score_separate(cand, gold, icd, strategy=strategy)
                    return gold_result
                elif match_type == "icd":
                    _, icd_result = match_score_separate(cand, gold, icd, strategy=strategy)
                    return icd_result
                else:
                    return match_score(cand, gold, icd, strategy=strategy)
            
            for r, cand in enumerate(pred_rankings or [], 1):
                # Find best score against ANY gold diagnosis
                best_cand_s = 0.0
                best_cand_tag = "negative"
                
                for g_item, i_item in zip(gold_list, icd_list):
                    s_curr, tag_curr = get_match_score(cand, g_item, i_item, strategy="semantic")
                    if s_curr > best_cand_s:
                        best_cand_s = s_curr
                        best_cand_tag = tag_curr # semantic tag is usually same ("semantic") but score matters
                
                sims.append(f"{best_cand_s:.4f}")
                w = log_rank_weight(r)
                weighted_scores.append(f"{(w * best_cand_s):.4f}")

                # For layered tag (exact/core found?), check all as well
                for g_item, i_item in zip(gold_list, icd_list):
                    _, tag_curr_layered = get_match_score(cand, g_item, i_item, strategy="layered")
                    if tag_curr_layered in ("exact", "core") and not exact_found:
                        exact_found = True

                if best_cand_s > best_s:
                    best_s = best_cand_s
                    best_pred = cand
                    best_rank = r
                    
                    # Find the layered tag for this best matching candidate
                    best_tag_layered = "none"
                    max_s_layered = -1.0
                    for g_item, i_item in zip(gold_list, icd_list):
                        s_lay, tag_lay = get_match_score(cand, g_item, i_item, strategy="layered")
                        if s_lay > max_s_layered:
                            max_s_layered = s_lay
                            best_tag_layered = tag_lay
                    best_tag = best_tag_layered

            # Top1 Score (against ANY gold)
            s1 = 0.0
            t1 = "none"
            if pred_top1:
                 for g_item, i_item in zip(gold_list, icd_list):
                    s_curr, t_curr = get_match_score(pred_top1, g_item, i_item, strategy="semantic")
                    if s_curr > s1:
                        s1 = s_curr
                        t1 = t_curr

            writer.writerow({
                'id': sample_id,
                'eval_diagnosis_name': gold_display,
                'icd_disease_name': icd_display,
                'pred_top1': pred_top1,
                'pred_top2': pred_top2,
                'pred_top3': pred_top3,
                'pred_full_rankings': pred_full,
                'similarities_per_pred': " | ".join(sims),
                'scores_per_pred': " | ".join(weighted_scores),
                'similarity_score_top1': f"{s1:.4f}",
                'top1_match_type': t1,
                'similarity_score_max': f"{best_s:.4f}",
                'best_match_pred': best_pred,
                'best_match_rank': best_rank,
                'best_match_type': best_tag,
                'exact_match_found': exact_found,
            })

    print(f"Generated debug CSV file: {csv_path}")
    print(f"CSV contains {len(gold_ids)} samples for model: {model_name}")

def generate_all_debug_csvs(
    out_dir: str,
    model_name: str,
    gold_ids: List[str],
    diagnosis_gold_ranked_lists: List[List[str]],
    diagnosis_icd_ranked_lists: List[List[str]],
    diagnosis_pred_rankings: List[List[str]],
    eval_data: List[Dict[str, Any]]
) -> None:
    """Generate three separate CSV files for different matching types"""
    print("Generating three separate CSV files...")
    
    # Generate CSV with max similarity (original behavior)
    generate_debug_csv_with_match_type(
        out_dir, model_name, gold_ids, diagnosis_gold_ranked_lists, 
        diagnosis_icd_ranked_lists, diagnosis_pred_rankings, eval_data, 
        match_type="max"
    )
    
    # Generate CSV with only original diagnosis name matching
    generate_debug_csv_with_match_type(
        out_dir, model_name, gold_ids, diagnosis_gold_ranked_lists, 
        diagnosis_icd_ranked_lists, diagnosis_pred_rankings, eval_data, 
        match_type="gold"
    )
    
    # Generate CSV with only ICD disease name matching
    generate_debug_csv_with_match_type(
        out_dir, model_name, gold_ids, diagnosis_gold_ranked_lists, 
        diagnosis_icd_ranked_lists, diagnosis_pred_rankings, eval_data, 
        match_type="icd"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["semantic", "layered"], default="semantic",
                        help="evaluation strategy for match_score (default: semantic)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated list of model names to evaluate. Use 'all' to process all models in the answers directory.")
    args = parser.parse_args()
    strategy = args.strategy

    cfg_paths = read_yaml(os.path.join("config", "paths.yml"))
    cfg_weights = read_yaml(os.path.join("config", "weights.yml"))
    cfg_api = read_yaml(os.path.join("config", "eval_llm_api.yml"))

    out_dir = cfg_paths.get("output_paths", {}).get(
        "llms_evaluation_results", os.path.join("results", "llms_evaluation_results")
    )
    os.makedirs(out_dir, exist_ok=True)

    # 仅保留最终诊断评估数据
    eval4_paths = cfg_paths.get("llm_eval4_data_paths", {})
    diag_eval = eval4_paths.get("final_diagnosis_eval")
    if not diag_eval or not os.path.exists(diag_eval):
        raise RuntimeError("paths.yml 缺少 final_diagnosis_eval 或文件不存在。请先生成评估数据。")

    def load_json_by_path(p: str) -> List[Dict[str, Any]]:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    data = load_json_by_path(diag_eval)

    # 汇总金标准（不再 normalize，直接原样）
    diagnosis_gold_primary: List[str] = []
    diagnosis_gold_codes: List[str] = []
    diagnosis_icd_names: List[str] = []
    # NEW: 多诊断（有序）列表
    diagnosis_gold_ranked_lists: List[List[str]] = []
    diagnosis_icd_ranked_lists: List[List[str]] = []
    gold_ids: List[str] = []
    for idx, item in enumerate(data, start=1):
        sid = item.get("id")
        if sid is None:
            sid = f"idx_{idx}"
        gold_ids.append(str(sid))
        eo = (item.get("expected_output") or {})
        diag_list = eo.get("诊断")

        # -------------------------------------------------------------------------
        # 1. 提取所有金标准诊断列表 (支持多诊断)
        # diagnosis_gold_ranked_lists: [[诊断A, 诊断B], [诊断C], ...]
        # -------------------------------------------------------------------------
        if isinstance(diag_list, list) and diag_list:
            ranked_names: List[str] = []
            ranked_icds: List[str] = []
            for d in diag_list:
                d = d if isinstance(d, dict) else {}
                ranked_names.append(d.get("诊断名称", "") or "")
                ranked_icds.append(d.get("ICD疾病名称", "") or "")

            diagnosis_gold_ranked_lists.append(ranked_names)
            diagnosis_icd_ranked_lists.append(ranked_icds)

            # 兼容旧的 primary 指标 (仅取第一个作为主诊断)
            name = ranked_names[0] if ranked_names else ""
            icd_name = ranked_icds[0] if ranked_icds else ""
            diagnosis_gold_primary.append(name)
            diagnosis_gold_codes.append(name)
            diagnosis_icd_names.append(icd_name)
        else:
            # -------------------------------------------------------------------------
            # 2. 单诊断数据格式回退处理
            # -------------------------------------------------------------------------
            pd = (eo.get("primary_diagnosis") or "")
            dc = (eo.get("diagnosis_code") or "") or pd
            icd_name = (eo.get("icd_disease_name") or "")
            diagnosis_gold_primary.append(pd)
            diagnosis_gold_codes.append(dc)
            diagnosis_icd_names.append(icd_name)
            # ranked fallback (包装成单元素列表)
            diagnosis_gold_ranked_lists.append([pd] if pd else [])
            diagnosis_icd_ranked_lists.append([icd_name] if icd_name else [])

    # 读取模型最终诊断答案（若默认目录缺失，则自动回退到 model_name+rag）
    answers_base_dir = cfg_paths.get(
        "llms_answers_data_base_dir", os.path.join("data", "llms_answers_data")
    )

    # Determine model list
    if args.models == "all":
        model_list = []
        # Support nested directory structure as seen in find_by_name results (e.g. Qwen/Qwen3-14B)
        for root, dirs, files in os.walk(answers_base_dir):
            if "final_diagnosis_pred.json" in files:
                # Use relative path from answers_base_dir as model name
                rel_path = os.path.relpath(root, answers_base_dir)
                if rel_path != ".":
                    model_list.append(rel_path)
        print(f"Found {len(model_list)} models in {answers_base_dir}")
    elif args.models:
        model_list = [m.strip() for m in args.models.split(',') if m.strip()]
    else:
        model_list = [cfg_api.get("model_name", "model")]

    def load_json_file(dir_path: str, file_name: str) -> List[Any]:
        p = os.path.join(dir_path, file_name)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    for model_name in model_list:
        answers_dir = os.path.join(answers_base_dir, model_name)
        alt_answers_dir = os.path.join(answers_base_dir, f"{model_name}+rag")

        used_model_name = model_name
        alt_final = load_json_file(alt_answers_dir, "final_diagnosis_pred.json")
        if alt_final:
            final_diag = alt_final
            used_model_name = f"{model_name}+rag"
        else:
            final_diag = load_json_file(answers_dir, "final_diagnosis_pred.json")
            if not final_diag:
                print(f"Warning: No answers found for model {model_name} in {answers_dir} or {alt_answers_dir}")
                continue

        safe_model_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(used_model_name))
        file_suffix = f"{safe_model_name}-{strategy}"

        # -------------------------------------------------------------------------
        # 3. 匹配模型预测结果到金标准ID
        # -------------------------------------------------------------------------
        pred_by_id: Dict[str, List[str]] = {}

        has_id = all(isinstance(item, dict) and ("id" in item) for item in final_diag) if final_diag else False
        if has_id:
            for item in final_diag:
                sid = str(item.get("id"))
                rks = list((item.get("rankings", []) or []))
                pred_by_id[sid] = rks
                # Also store with cleaned ID (remove trailing periods) for better matching
                sid_clean = sid.rstrip('。.')
                if sid_clean != sid:
                    pred_by_id[sid_clean] = rks
            
            diagnosis_pred_rankings = []
            for sid in gold_ids:
                # Try exact match first, then try cleaned version
                rankings = pred_by_id.get(sid, [])
                if not rankings:
                    sid_clean = sid.rstrip('。.')
                    rankings = pred_by_id.get(sid_clean, [])
                diagnosis_pred_rankings.append(rankings)
            
            # Print matching statistics for verification
            matched_count = sum(1 for rankings in diagnosis_pred_rankings if rankings)
            print(f"Matched {matched_count}/{len(gold_ids)} predictions with evaluation data for model {model_name}")
        else:
            diagnosis_pred_rankings = [
                list((item.get("rankings", []) or []))
                for item in final_diag
            ]
            if len(diagnosis_pred_rankings) != len(gold_ids):
                diagnosis_pred_rankings = diagnosis_pred_rankings[:len(gold_ids)]

        diagnosis_pred_codes = [
            (rks[0] if isinstance(rks, list) and len(rks) > 0 else "")
            for rks in diagnosis_pred_rankings
        ]

        # 计算最终诊断指标（统一策略接口：semantic / layered）
        # 原始的综合评估（取最佳匹配）
        diag_metrics = evaluate_final_diagnosis(
            diagnosis_pred_rankings,
            diagnosis_gold_primary,
            icd_names=diagnosis_icd_names,
            strategy=strategy,
        )

        # NEW: 多诊断（有序）评估：加权 nDCG@K + 噪声惩罚 FinalScore
        ranked_metrics = evaluate_ranked_diagnoses(
            diagnosis_pred_rankings,
            diagnosis_gold_ranked_lists,
            gold_icd_ranked_lists=diagnosis_icd_ranked_lists,
            K=int(cfg_weights.get("ndcg_k", 10)),
            lambda_=float(cfg_weights.get("ndcg_lambda", 0.7)),
            rel_threshold=float(cfg_weights.get("rel_threshold", 0.7)),
            beta=float(cfg_weights.get("precision_beta", 0.7)),
            strategy="semantic",  # rel 使用连续分数更合适
        )

        # 分别评估原始诊断名称和ICD疾病名称的匹配结果
        diag_metrics_separate = evaluate_final_diagnosis_separate(
            diagnosis_pred_rankings,
            diagnosis_gold_primary,
            icd_names=diagnosis_icd_names,
            strategy=strategy,
        )

        # 汇总分
        summary: Dict[str, Any] = {}
        # NEW: 总分使用多诊断排序 FinalScore（summarize 会优先读 final_score）
        if ranked_metrics:
            summary.update(summarize_diag_metrics(ranked_metrics, cfg_weights))
        elif diag_metrics:
            summary.update(summarize_diag_metrics({**diag_metrics}, cfg_weights))

        score_total = float(summary.get("Score_diag", 0.0))

        # Construct report with requested structure
        report: Dict[str, Any] = {
            "model_name": used_model_name,
            "summary_scores": {
                "Score_total": score_total
            },
            "ranked_multi_diag": ranked_metrics or {},
            "combined": {},
            "gold": {},
            "icd": {}
        }
        
        # Keys to include in combined, gold, and icd sections
        metric_keys = [
            "top1_accuracy", "top2_accuracy", "top3_accuracy",
            "mrr_primary", "precision", "recall", "f1_score",
            "avg_weighted_score", "weighted_semantic_mean"
        ]

        if diag_metrics:
            for k in metric_keys:
                if k in diag_metrics:
                    report["combined"][k] = diag_metrics[k]

        if diag_metrics_separate:
            for k in metric_keys:
                gold_k = f"gold_{k}"
                icd_k = f"icd_{k}"
                if gold_k in diag_metrics_separate:
                    report["gold"][k] = diag_metrics_separate[gold_k]
                if icd_k in diag_metrics_separate:
                    report["icd"][k] = diag_metrics_separate[icd_k]

        out_path = os.path.join(out_dir, f"llm_evaluation_metrics_{file_suffix}.json")
        save_json(out_path, report)
        print(f"Saved LLM evaluation metrics to {out_path} for model {model_name}")

        # 图表：Top‑k 柱状图
        fig_dir = os.path.join(out_dir, f"figures_{file_suffix}")
        try:
            plot_topk_accuracy(diag_metrics or {}, fig_dir, file_suffix)
            print(f"Saved figures to {fig_dir} for model {model_name}")
        except Exception:
            pass

        # CSV 调试 - Generate three separate CSV files
        try:
            generate_all_debug_csvs(
                out_dir,
                file_suffix,
                gold_ids,
                diagnosis_gold_ranked_lists,
                diagnosis_icd_ranked_lists,
                diagnosis_pred_rankings,
                data,
            )
        except Exception as e:
            print(f"Warning: Failed to generate debug CSV for model {model_name}: {e}")

if __name__ == "__main__":
    main()
