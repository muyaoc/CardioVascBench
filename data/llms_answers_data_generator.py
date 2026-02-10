#!/usr/bin/env python3
"""
基于大模型 API 的真实答案生成器（仅保留“最终诊断”任务）

功能：
- 读取 config/eval_llm_api.yml（api_url/api_key/model_name）
- 读取 config/paths.yml 的 llm_eval4_data_paths.final_diagnosis_eval 作为提问数据源
- 针对每个样本构造提示词，请模型按指定 JSON 模式返回答案
- 将答案保存到 data/llms_answers_data/<model_name>/final_diagnosis_pred.json（按样本 id 对齐，输出包含 id 与 rankings）
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import yaml

try:
    import requests
except Exception:
    requests = None


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_error_log(path: str, message: str) -> None:
    """追加错误日志到指定路径，含时间戳。"""
    try:
        ensure_dir(os.path.dirname(path))
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")
    except Exception:
        # 日志写入失败不影响主流程
        pass


def _dump_mr(medical_record: Optional[Dict[str, Any]]) -> str:
    if isinstance(medical_record, dict) and medical_record:
        try:
            return json.dumps(medical_record, ensure_ascii=False)
        except Exception:
            return str(medical_record)
    return ""


def build_system_prompt() -> str:
    return (
        "你是一名心血管内科的临床医生，负责根据给定的病历文书和心电检查结果，"
        "生成结构化的最终诊断。请严格输出 JSON 对象，"
        "不要输出多余文本。中文输出，字段与取值尽量标准化。"
    )


def build_final_diag_prompt(medical_record: Optional[Dict[str, Any]], ecg_results: List[Dict[str, Any]]) -> str:
    mr_text = _dump_mr(medical_record)
    try:
        ecg_text = json.dumps(ecg_results, ensure_ascii=False)
    except Exception:
        ecg_text = str(ecg_results)
    return (
        "任务：最终诊断。输入病历文书和心电检查项目、所见、诊断，输出最终诊断。\n"
        + (f"病历文书（摘要）：{mr_text}\n" if mr_text else "")
        + f"ECG结果：{ecg_text}\n\n"
        + "请仅返回 JSON，包含这些键：\n"
        + "- 诊断: 数组，每项为对象，仅包含键 '诊断名称'（规范中文主诊断）\n\n"
        + "严格要求：\n"
        + "1) 返回合法 JSON 对象，键名与类型符合要求\n"
        + "2) 不要加入解释或额外字段\n"
    )

# === 新增：RAG 集成辅助函数 ===
try:
    # 允许直接通过 src 目录导入 RAG 客户端
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
    from rag_client import create_rag_client, RetrievalResult  # type: ignore
except Exception:
    # 内置后备：直接调用 RAG API 的轻量客户端
    class _RetrievalResult:
        def __init__(self, content: str):
            self.content = content
    RetrievalResult = _RetrievalResult  # type: ignore

    def _load_rag_cfg(path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    class _DirectRAGClient:
        def __init__(self, cfg: Dict[str, Any]):
            self.base = str(cfg.get("api_url", "")).rstrip("/")
            self.api_key = str(cfg.get("api_key", ""))
            self.retrieval = cfg.get("retrieval_config", {}) or {}
            self.advanced = cfg.get("advanced_config", {}) or {}
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            # 复用全局 requests；若缺失则抛异常由上层处理
            if requests is None:
                raise RuntimeError("缺少 requests 依赖，无法启用 RAG 直连客户端")
            self.session = requests.Session()

        def health_check(self) -> bool:
            try:
                url = self.base + "/retrieve"
                payload = {
                    "query": "ping",
                    "top_k": 1,
                    "search_mode": self.retrieval.get("search_mode", "semantic"),
                }
                resp = self.session.post(url, headers=self.headers, json=payload, timeout=10)
                return getattr(resp, "status_code", 500) == 200
            except Exception:
                return False

        def retrieve(self, query: str):
            try:
                url = self.base + "/retrieve"
                payload: Dict[str, Any] = {
                    "query": str(query or "")[:500],
                    "top_k": int(self.retrieval.get("top_k", 5)),
                    "search_mode": self.retrieval.get("search_mode", "semantic"),
                    "similarity_threshold": float(self.retrieval.get("similarity_threshold", 0.0)),
                }
                if self.advanced:
                    payload["advanced_config"] = self.advanced
                resp = self.session.post(url, headers=self.headers, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                results = []
                items = data if isinstance(data, list) else (data.get("results") or [])
                for r in items:
                    try:
                        content = (r.get("content") or r.get("text") or r.get("chunk") or "").strip()
                    except Exception:
                        content = ""
                    if content:
                        results.append(_RetrievalResult(content))
                return results
            except Exception:
                return []

    def create_rag_client(cfg_path: str):  # type: ignore
        cfg = _load_rag_cfg(cfg_path)
        return _DirectRAGClient(cfg)

def build_rag_retrieval_query(medical_record: Optional[Dict[str, Any]], ecg_results: List[Dict[str, Any]]) -> str:
    """根据病历与心电结果构造简洁的检索查询，避免 422。"""
    parts: List[str] = []
    if isinstance(medical_record, dict):
        # 选取若干关键栏位简要串联
        for k in ("chief_complaint", "present_illness", "past_history", "diagnosis_summary"):
            v = medical_record.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
    if isinstance(ecg_results, list):
        for e in ecg_results[:5]:
            if isinstance(e, dict):
                name = (e.get("name") or e.get("项目") or "").strip()
                diag = (e.get("diagnosis") or e.get("诊断") or "").strip()
                obs = (e.get("observations") or e.get("所见") or "").strip()
                s = ", ".join([x for x in (name, obs, diag) if x])
                if s:
                    parts.append(s)
    query = "；".join(parts)
    # 限制长度，保障服务端可接受
    return query[:300]

def make_rag_context(rag_client, query: str, max_items: int = 5, max_chars: int = 1500) -> str:
    """调用 RAG 检索并拼装简短参考上下文。失败返回空字符串。"""
    if not rag_client or not query:
        return ""
    try:
        results: List[RetrievalResult] = rag_client.retrieve(query)
        if not results:
            return ""
        lines: List[str] = []
        for r in results[:max_items]:
            content = (getattr(r, "content", "") or "").strip()
            if content:
                # 简单清洗过长段落
                if len(content) > 400:
                    content = content[:400] + "..."
                lines.append(f"- {content}")
        ctx = "\n".join(lines)
        if len(ctx) > max_chars:
            ctx = ctx[:max_chars] + "..."
        return ctx
    except Exception as e:
        return ""


def call_chat_completions(api_url: str, api_key: str, model_name: str, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
    if requests is None:
        raise RuntimeError("缺少 requests 依赖，请先安装：pip install requests")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }
    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )
        if not content:
            return None
        return json.loads(content)
    except Exception:
        return None


def test_api_connectivity(api_url: str, api_key: str, model_name: str) -> bool:
    """轻量测试 API 是否可用，返回 True/False。"""
    if requests is None:
        return False
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "你是一个助手。"},
            {"role": "user", "content": "ping"},
        ],
        "temperature": 0,
    }
    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # 只要返回结构包含 choices 即认为连通
        return bool(data.get("choices"))
    except Exception:
        return False


def call_chat_completions_with_retry(
    api_url: str,
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    error_log_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """带重试与指数回退的请求封装，失败时记录错误日志。"""
    if requests is None:
        append_error_log(error_log_path or "", "requests 未安装，无法发起 API 请求")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload_base = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(api_url, headers=headers, json=payload_base, timeout=60)
            status = getattr(resp, "status_code", None)
            resp.raise_for_status()
            data = resp.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content")
            )
            if not content:
                raise ValueError("API返回内容为空或缺少message.content")
            return json.loads(content)
        except Exception as e:
            # 记录错误
            if error_log_path:
                append_error_log(error_log_path, f"请求失败 第{attempt}/{max_retries}次：{type(e).__name__}: {e}")
            # 指数回退等待（最后一次失败不再等待）
            if attempt < max_retries:
                sleep_s = backoff_seconds * (2 ** (attempt - 1))
                try:
                    time.sleep(sleep_s)
                except Exception:
                    pass
            else:
                break
    return None


def coerce_answer(ans: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(ans, dict):
        diag_entries = ans.get("诊断")
        diag_names: List[str] = []
        if isinstance(diag_entries, list):
            for d in diag_entries:
                if isinstance(d, dict):
                    name = (d.get("诊断名称") or "").strip()
                    if name:
                        diag_names.append(name)
                elif isinstance(d, (str, int, float)):
                    name = str(d).strip()
                    if name:
                        diag_names.append(name)
        return {
            "diagnosis_pred_rankings": diag_names,
            "diagnosis_pred_codes": diag_names[0] if diag_names else "",
        }
    return {
        "diagnosis_pred_rankings": [],
        "diagnosis_pred_codes": "",
    }


def main() -> None:
    cfg_api = read_yaml(os.path.join("config", "eval_llm_api.yml"))
    api_url = cfg_api.get("api_url")
    api_key = cfg_api.get("api_key")
    model_name = cfg_api.get("model_name", "model")
    if not api_url or not api_key:
        raise RuntimeError("请在 config/eval_llm_api.yml 中配置 api_url 与 api_key。")

    # # 在生成前测试 API 连通性并打印通断
    # connected = test_api_connectivity(api_url, api_key, model_name)
    # print(f"API连通性：{'通' if connected else '断'}")
    # if not connected:
    #     raise RuntimeError("API连通性：断，已中止执行。")

    # 尝试加载 RAG 配置并做健康检查
    rag_client = None
    rag_enabled = False
    try:
        rag_cfg_path = os.path.join("config", "eval_rag_api.yml")
        if os.path.exists(rag_cfg_path) and create_rag_client:
            rag_client = create_rag_client(rag_cfg_path)
            try:
                rag_enabled = bool(rag_client) and rag_client.health_check()
            except Exception:
                rag_enabled = False
    except Exception:
        rag_enabled = False
    print(f"RAG：{'启用' if rag_enabled else '未启用'}")

    cfg_paths = read_yaml(os.path.join("config", "paths.yml"))
    eval4_paths = cfg_paths.get("llm_eval4_data_paths", {})
    diag_eval_path = eval4_paths.get("final_diagnosis_eval")
    if not diag_eval_path or not os.path.exists(diag_eval_path):
        raise RuntimeError("未找到 final_diagnosis_eval，请先运行 llms_eval_data_generator.py 生成评估数据。")

    # 结果目录按是否启用 RAG 增加后缀，以便与评估脚本配置一致
    answers_base_dir = cfg_paths.get("llms_answers_data_base_dir", os.path.join("data", "llms_answers_data"))
    model_name_out = f"{model_name}+rag" if rag_enabled else model_name
    answers_base = os.path.join(answers_base_dir, model_name_out)
    ensure_dir(answers_base)
    errors_log_path = os.path.join(answers_base, "api_errors.log")

    diagnosis_pred_rankings: List[List[str]] = []
    diagnosis_pred_codes: List[str] = []
    sample_ids: List[str] = []
    fail_count: int = 0

    sys_prompt = build_system_prompt()

    try:
        with open(diag_eval_path, "r", encoding="utf-8") as f:
            diag_data = json.load(f)
        total = len(diag_data) if isinstance(diag_data, list) else 0
        if isinstance(diag_data, list):
            for idx, item in enumerate(diag_data, start=1):
                inp = item.get("input") or {}
                mr = inp.get("medical_record") if isinstance(inp.get("medical_record"), dict) else None
                ecg_results = inp.get("ecg_results")
                if not isinstance(ecg_results, list):
                    ecg_results = []
                # 记录样本 id（若缺失则使用索引占位）
                sid = item.get("id")
                if sid is None:
                    sid = f"idx_{idx}"
                sample_ids.append(str(sid))
                # 构造用户提示，按需注入 RAG 检索到的参考资料
                user_prompt = build_final_diag_prompt(mr, ecg_results)
                if rag_enabled and rag_client:
                    query = build_rag_retrieval_query(mr, ecg_results)
                    rag_ctx = make_rag_context(rag_client, query)
                    if rag_ctx:
                        # 将参考资料附加到提示末尾，避免破坏主任务结构
                        user_prompt = (
                            user_prompt
                            + "\n参考资料（RAG检索摘要，仅供参考）：\n"
                            + rag_ctx
                        )
                    else:
                        append_error_log(errors_log_path, f"RAG检索为空或失败：sample_id={sid}")
                ans = call_chat_completions_with_retry(
                    api_url,
                    api_key,
                    model_name,
                    sys_prompt,
                    user_prompt,
                    max_retries=3,
                    backoff_seconds=1.0,
                    error_log_path=errors_log_path,
                )
                parsed = coerce_answer(ans)
                diagnosis_pred_rankings.append(parsed.get("diagnosis_pred_rankings", []))
                diagnosis_pred_codes.append(parsed.get("diagnosis_pred_codes", ""))
                if ans is None:
                    fail_count += 1
                # 进度打印：当前/总数
                if total:
                    print(f"生成中：{idx}/{total}（失败{fail_count}）", end="\r", flush=True)
                time.sleep(0.2)
    except Exception:
        pass
    finally:
        # 打印最终完成进度并换行
        try:
            if 'total' in locals() and total:
                print(f"生成完成：{len(diagnosis_pred_rankings)}/{total}（失败{fail_count}）")
        except Exception:
            # 安全兜底，避免因为打印出错影响后续保存
            pass

    # 保存最终诊断答案文件（仅 rankings）
    final_diag_combined: List[Dict[str, Any]] = []
    # 按采集到的 sample_ids 对齐输出，确保包含 id
    for i in range(len(sample_ids)):
        final_diag_combined.append({
            "id": sample_ids[i],
            "rankings": diagnosis_pred_rankings[i] if i < len(diagnosis_pred_rankings) else [],
        })
    save_json(os.path.join(answers_base, "final_diagnosis_pred.json"), final_diag_combined)

    print(f"LLM final diagnosis answers generated via API for model '{model_name_out}'.")


if __name__ == "__main__":
    main()
