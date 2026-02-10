#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import requests
import yaml

DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0


# ---------- utils ----------
def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_error_log(path: str, message: str) -> None:
    ensure_dir(os.path.dirname(path))
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


def normalize_base_url(u: str) -> str:
    """
    Normalize base URL so callers can pass either:
      - http://host:port
      - http://host:port/
      - http://host:port/v1
      - http://host:port/v1/
    and we always store:
      - http://host:port
    """
    u = (u or "").strip().rstrip("/")
    if u.endswith("/v1"):
        u = u[:-3]
    return u


ParsedJSON = Union[Dict[str, Any], List[Any]]


def _try_parse_python_literal(text: str) -> Optional[ParsedJSON]:
    """
    Best-effort parse for python-literal-like outputs, e.g.
      {'诊断名称': '冠心病'}
      [{'diagnosisName': '心绞痛'}]
    """
    if not text:
        return None
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        return None
    return None


def try_parse_json_from_text(text: str) -> Optional[ParsedJSON]:
    """
    Try hard to extract a JSON object/array from model output.
    Accepts:
      - pure JSON
      - fenced ```json ... ```
      - text that contains an embedded JSON object/array
      - python-literal-like dict/list (single quotes) via ast.literal_eval
    Returns a Python dict OR list if successful; otherwise None.
    """
    if not text:
        return None

    # 1) strict JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass

    # 1.5) python literal (single quotes)
    lit = _try_parse_python_literal(text)
    if lit is not None:
        return lit

    # 2) fenced
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        inner = m.group(1).strip()

        # 2a) strict JSON in fence
        try:
            obj = json.loads(inner)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            pass

        # 2b) python literal in fence
        lit2 = _try_parse_python_literal(inner)
        if lit2 is not None:
            return lit2

        # keep scanning within inner
        text = inner

    # 3) scan for first JSON object/array
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        if text[i] in "{[":
            try:
                obj, _ = dec.raw_decode(text[i:])
                if isinstance(obj, (dict, list)):
                    return obj
            except Exception:
                pass
        i += 1

    # 3.5) last attempt: python literal scan
    for pat in (r"(\{.*\})", r"(\[.*\])"):
        mm = re.search(pat, text, flags=re.S)
        if mm:
            lit3 = _try_parse_python_literal(mm.group(1).strip())
            if lit3 is not None:
                return lit3

    return None


# very small template renderer for patterns like:
# {% for m in messages %}...{% if m.role=='user' %}...{{ m.content }}...{% endif %}{% endfor %}
def render_template(tmpl: str, messages: List[Dict[str, str]]) -> str:
    if "{% for m in messages %}" not in tmpl:
        return tmpl

    pre, rest = tmpl.split("{% for m in messages %}", 1)
    body, post = rest.split("{% endfor %}", 1)

    out = [pre]
    for m in messages:
        seg = body
        role = m.get("role", "")
        # Treat "developer" like "system" for template routing
        role_for_tpl = "system" if role == "developer" else role
        content = (m.get("content") or "").strip()

        # handle {% if ... %} branches (system/user/assistant)
        def choose(s: str) -> str:
            if "{% if m.role=='system' %}" in s:
                a, s2 = s.split("{% if m.role=='system' %}", 1)
                sys_blk, s2 = s2.split("{% elif m.role=='user' %}", 1)
                usr_blk, s2 = s2.split("{% elif m.role=='assistant' %}", 1)
                as_blk, s2 = s2.split("{% endif %}", 1)
                chosen = (
                    sys_blk
                    if role_for_tpl == "system"
                    else usr_blk
                    if role_for_tpl == "user"
                    else as_blk
                    if role_for_tpl == "assistant"
                    else ""
                )
                return a + chosen + s2

            if "{% if m.role=='user' %}" in s and "{% elif m.role=='assistant' %}" in s:
                a, s2 = s.split("{% if m.role=='user' %}", 1)
                usr_blk, s2 = s2.split("{% elif m.role=='assistant' %}", 1)
                as_blk, s2 = s2.split("{% endif %}", 1)
                chosen = usr_blk if role_for_tpl == "user" else as_blk if role_for_tpl == "assistant" else ""
                return a + chosen + s2

            return s

        seg = choose(seg).replace("{{ m.content }}", content)
        out.append(seg)

    out.append(post)

    rendered = "".join(out)
    rendered = re.sub(r"\{\%.*?\%\}", "", rendered, flags=re.S)
    return rendered


def build_system_prompt() -> str:
    return (
        "你是一名心血管内科的临床医生，负责根据给定的病历文书和心电检查结果，"
        "生成结构化的最终诊断。请严格输出 JSON 对象，"
        "不要输出多余文本。中文输出，字段与取值尽量标准化。"
    )


def _dump_mr(mr: Optional[Dict[str, Any]]) -> str:
    if isinstance(mr, dict) and mr:
        try:
            return json.dumps(mr, ensure_ascii=False)
        except Exception:
            return str(mr)
    return ""


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


def coerce_answer(ans: Any) -> Dict[str, Any]:
    """
    Robustly normalize many JSON shapes into:
      {"diagnosis_pred_rankings": [...], "diagnosis_pred_codes": first_or_empty}
    """
    NAME_KEYS = (
        "诊断名称",
        "诊断名",
        "主诊断",
        "主要诊断",
        "诊断",
        "最终诊断",
        "diagnosisName",
        "diagnosis_name",
        "diagnosis name",
        "diagnosis",
        "final_diagnosis",
        "finalDiagnosis",
        "classification",
        "label",
        "prediction",
        "pred",
        "result",
        "name",
        "Name",
        "title",
        "Title",
    )

    WRAP_KEYS = (
        "诊断",
        "diagnosis",
        "final_diagnosis",
        "finalDiagnosis",
        "outputs",
        "output",
        "data",
        "answer",
        "answers",
        "response",
    )

    def norm_str(x: Any) -> Optional[str]:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            s = str(x).strip()
            return s or None
        if isinstance(x, str):
            s = x.strip()
            if s in ("```", "```json", "```JSON"):
                return None
            s = s.strip("`").strip()
            if not s or s.lower() in ("null", "none", "n/a"):
                return None
            return s
        return None

    def extend_from_value(out: List[str], v: Any) -> None:
        if v is None:
            return
        s = norm_str(v)
        if s is not None:
            out.append(s)
            return
        if isinstance(v, dict):
            out.extend(collect_names(v))
            return
        if isinstance(v, list):
            for it in v:
                extend_from_value(out, it)
            return

    def collect_names(obj: Any) -> List[str]:
        out: List[str] = []
        if obj is None:
            return out

        if isinstance(obj, dict) and "diagnosis_pred_rankings" in obj:
            v = obj.get("diagnosis_pred_rankings")
            extend_from_value(out, v)
            return out

        s = norm_str(obj)
        if s is not None:
            out.append(s)
            return out

        if isinstance(obj, list):
            for it in obj:
                out.extend(collect_names(it))
            return out

        if isinstance(obj, dict):
            for k in WRAP_KEYS:
                if k in obj and obj[k] is not None:
                    extend_from_value(out, obj[k])
                    if out:
                        return out

            for k in NAME_KEYS:
                if k in obj and obj[k] is not None:
                    extend_from_value(out, obj[k])
            if out:
                return out

            for k, v in obj.items():
                if isinstance(k, str):
                    if re.fullmatch(r"诊断\d+", k) or re.fullmatch(r"diagnosis\d+", k, flags=re.I):
                        extend_from_value(out, v)
            if out:
                return out

            for k, v in obj.items():
                if not isinstance(k, str):
                    continue
                kk = k.lower()
                if ("诊断" in k) or ("diagnosis" in kk):
                    extend_from_value(out, v)
            if out:
                return out

            for v in obj.values():
                s2 = norm_str(v)
                if s2 is not None:
                    out.append(s2)

            return out

        return out

    names = collect_names(ans)

    seen = set()
    deduped: List[str] = []
    for n in names:
        nn = (n or "").strip()
        if not nn:
            continue
        if nn in ("{}", "[]", "```"):
            continue
        if nn not in seen:
            seen.add(nn)
            deduped.append(nn)

    return {
        "diagnosis_pred_rankings": deduped,
        "diagnosis_pred_codes": deduped[0] if deduped else "",
    }


# ---------- compatibility layer ----------
def _safe_get_error(resp: requests.Response) -> Dict[str, Any]:
    """Best-effort extract {error:{...}} body."""
    try:
        j = resp.json()
        if isinstance(j, dict) and isinstance(j.get("error"), dict):
            return j["error"]
    except Exception:
        pass
    return {}


def _rename_key(d: Dict[str, Any], src: str, dst: str) -> Optional[Dict[str, Any]]:
    if src not in d:
        return None
    nd = dict(d)
    nd[dst] = nd.pop(src)
    return nd


def _drop_key(d: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    if key not in d:
        return None
    nd = dict(d)
    nd.pop(key, None)
    return nd


def _rewrite_system_role(payload: Dict[str, Any], new_role: str) -> Optional[Dict[str, Any]]:
    msgs = payload.get("messages")
    if not isinstance(msgs, list):
        return None
    changed = False
    new_payload = deepcopy(payload)
    for m in new_payload["messages"]:
        if isinstance(m, dict) and m.get("role") == "system":
            m["role"] = new_role
            changed = True
    return new_payload if changed else None


def _merge_system_into_user(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    msgs = payload.get("messages")
    if not isinstance(msgs, list):
        return None

    sys_parts: List[str] = []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "system" and (m.get("content") or "").strip():
            sys_parts.append((m.get("content") or "").strip())
    if not sys_parts:
        return None

    merged_sys = "\n\n".join(sys_parts).strip()
    new_payload = deepcopy(payload)
    new_msgs = [m for m in new_payload["messages"] if not (isinstance(m, dict) and m.get("role") == "system")]

    for m in new_msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            m["content"] = f"{merged_sys}\n\n{(m.get('content') or '').strip()}".strip()
            break
    else:
        new_msgs.insert(0, {"role": "user", "content": merged_sys})

    new_payload["messages"] = new_msgs
    return new_payload


def apply_compat_fix(
    payload: Dict[str, Any],
    err: Dict[str, Any],
    endpoint_kind: str,  # "chat" or "completions"
) -> Optional[tuple[Dict[str, Any], str]]:
    """
    Return (new_payload, action_name) if we can fix and retry.
    """
    msg = str(err.get("message") or "")
    param = str(err.get("param") or "")
    code = str(err.get("code") or "")

    # 1) max_tokens <-> max_completion_tokens
    if (param == "max_tokens" or "max_completion_tokens" in msg) and "max_tokens" in payload:
        nd = _rename_key(payload, "max_tokens", "max_completion_tokens")
        if nd:
            return nd, "max_tokens->max_completion_tokens"
    if (param == "max_completion_tokens" or "max_tokens" in msg) and "max_completion_tokens" in payload:
        nd = _rename_key(payload, "max_completion_tokens", "max_tokens")
        if nd:
            return nd, "max_completion_tokens->max_tokens"

    # 2) response_format not supported
    if param == "response_format" or ("response_format" in msg and code == "unsupported_parameter"):
        nd = _drop_key(payload, "response_format")
        if nd:
            return nd, "drop_response_format"

    # 3) stop / temperature not supported
    if param == "stop" or ("stop" in msg and code == "unsupported_parameter"):
        nd = _drop_key(payload, "stop")
        if nd:
            return nd, "drop_stop"
    if param == "temperature" or ("temperature" in msg and code == "unsupported_parameter"):
        nd = _drop_key(payload, "temperature")
        if nd:
            return nd, "drop_temperature"

    # 4) system role not supported -> try developer, then merge
    if endpoint_kind == "chat":
        # Common patterns: "Unsupported value: messages[0].role: system"
        if ("messages" in msg and "role" in msg and "system" in msg) or ("role" in msg and "system" in msg):
            nd = _rewrite_system_role(payload, "developer")
            if nd:
                return nd, "system->developer"
            nd2 = _merge_system_into_user(payload)
            if nd2:
                return nd2, "merge_system_into_user"

    return None


# ---------- HTTP ----------
def _log_http_error(errlog: str, k: int, where: str, e: requests.exceptions.HTTPError) -> None:
    resp = e.response
    if resp is None:
        append_error_log(errlog, f"请求失败({where}) 第{k}: HTTPError(no response): {e}")
        return
    append_error_log(errlog, f"请求失败({where}) 第{k}: HTTP {resp.status_code} body_snippet={resp.text[:1200]!r}")


def call_completions(
    session: requests.Session,
    base_url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: int,
    retries: int,
    backoff: float,
    errlog: str,
) -> Optional[ParsedJSON]:
    url = base_url.rstrip("/") + "/v1/completions"

    for k in range(1, retries + 1):
        try:
            cur_payload = payload
            tried_actions: set[str] = set()

            while True:
                r = session.post(url, headers=headers, json=cur_payload, timeout=timeout)

                if r.status_code == 400:
                    err = _safe_get_error(r)
                    fix = apply_compat_fix(cur_payload, err, endpoint_kind="completions")
                    if fix:
                        new_payload, action = fix
                        if action not in tried_actions:
                            tried_actions.add(action)
                            append_error_log(
                                errlog,
                                f"CompatFix(completions) 第{k}: {action} | param={err.get('param')} code={err.get('code')} msg={str(err.get('message') or '')[:300]!r}",
                            )
                            cur_payload = new_payload
                            continue  # retry immediately
                # no fix -> normal flow
                try:
                    r.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    _log_http_error(errlog, k, "completions", e)
                    raise

                data = r.json()
                content = (data.get("choices", [{}])[0].get("text") or "").strip()
                parsed = try_parse_json_from_text(content)
                if parsed is not None:
                    return parsed

                append_error_log(
                    errlog,
                    f"JSON解析失败(completions) 第{k}: content_snippet={content[:800]!r} | raw_data={str(data)[:1200]}",
                )
                break  # parsed failed: leave inner loop to outer retry

        except requests.exceptions.HTTPError:
            pass  # already logged with body
        except Exception as e:
            append_error_log(errlog, f"请求失败(completions) 第{k}: {type(e).__name__}: {e}")

        if k < retries:
            time.sleep(backoff * (2 ** (k - 1)))

    return None


def call_chat(
    session: requests.Session,
    base_url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: int,
    retries: int,
    backoff: float,
    errlog: str,
) -> Optional[ParsedJSON]:
    url = base_url.rstrip("/") + "/v1/chat/completions"

    for k in range(1, retries + 1):
        try:
            cur_payload = payload
            tried_actions: set[str] = set()

            while True:
                r = session.post(url, headers=headers, json=cur_payload, timeout=timeout)

                if r.status_code == 400:
                    err = _safe_get_error(r)
                    fix = apply_compat_fix(cur_payload, err, endpoint_kind="chat")
                    if fix:
                        new_payload, action = fix
                        if action not in tried_actions:
                            tried_actions.add(action)
                            append_error_log(
                                errlog,
                                f"CompatFix(chat) 第{k}: {action} | param={err.get('param')} code={err.get('code')} msg={str(err.get('message') or '')[:300]!r}",
                            )
                            cur_payload = new_payload
                            continue  # retry immediately

                try:
                    r.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    _log_http_error(errlog, k, "chat", e)
                    raise

                data = r.json()
                content = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
                parsed = try_parse_json_from_text(content)
                if parsed is not None:
                    return parsed

                append_error_log(
                    errlog,
                    f"JSON解析失败(chat) 第{k}: content_snippet={content[:800]!r} | raw_data={str(data)[:1200]}",
                )
                break  # parsed failed: leave inner loop to outer retry

        except requests.exceptions.HTTPError:
            pass  # already logged with body
        except Exception as e:
            append_error_log(errlog, f"请求失败(chat) 第{k}: {type(e).__name__}: {e}")

        if k < retries:
            time.sleep(backoff * (2 ** (k - 1)))

    return None


# ---------- main ----------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-cfg", required=True, help="Path to model YAML (contains both config and template)")
    ap.add_argument("--paths-cfg", required=True, help="Path to paths.yml")
    args = ap.parse_args()

    # API from environment (set by the .sh)
    api_url = normalize_base_url(os.getenv("API_URL", ""))
    api_key = (os.getenv("API_KEY", "") or "").strip()
    if not api_url or not api_key:
        raise SystemExit("Missing API_URL/API_KEY env vars (export them in your .sh).")

    mcfg = read_yaml(args.model_cfg)
    pcfg = read_yaml(args.paths_cfg)

    model_name = mcfg.get("model_name", "model")
    endpoint = (mcfg.get("endpoint") or "completions").lower()  # "completions" or "chat"
    use_client_tpl = bool(mcfg.get("use_client_templates", True))
    supports_system = bool(mcfg.get("supports_system_role", False))
    json_mode = bool(mcfg.get("json_mode", False))
    stops = mcfg.get("stops", []) or []

    max_tokens = int(mcfg.get("max_tokens", DEFAULT_MAX_TOKENS))
    token_param = str(mcfg.get("token_param") or "max_tokens").strip()  # compatibility knob
    temperature = float(mcfg.get("temperature", DEFAULT_TEMPERATURE))
    timeout_seconds = int(mcfg.get("timeout_seconds", 60))
    max_retries = int(mcfg.get("max_retries", 3))
    backoff_seconds = float(mcfg.get("backoff_seconds", 1.0))

    template = (mcfg.get("template") or "").strip()

    diag_eval_path = pcfg.get("final_diagnosis_eval")
    print(diag_eval_path)
    if not diag_eval_path or not os.path.exists(diag_eval_path):
        raise SystemExit("Missing final_diagnosis_eval path or file does not exist in paths.yml.")

    answers_base_dir = pcfg.get("llms_answers_data_base_dir", os.path.join("data", "llms_answers_data"))
    answers_dir = os.path.join(answers_base_dir, model_name)
    ensure_dir(answers_dir)
    errlog = os.path.join(answers_dir, "api_errors.log")

    s = requests.Session()
    s.mount("http://", requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=0))
    s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=0))

    # connectivity check
    try:
        r = s.get(api_url.rstrip("/") + "/v1/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        print("API连通性：通" if r.ok else "API连通性：断")
        if not r.ok:
            append_error_log(errlog, f"Connectivity failed: HTTP {r.status_code} body={r.text[:1200]!r}")
            raise SystemExit("Connectivity failed.")
    except Exception as e:
        print("API连通性：断")
        append_error_log(errlog, f"Connectivity failed: {type(e).__name__}: {e}")
        raise SystemExit("Connectivity failed.")

    with open(diag_eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    total = len(eval_data) if isinstance(eval_data, list) else 0

    sys_prompt = build_system_prompt()

    ids: List[str] = []
    rankings: List[List[str]] = []
    codes: List[str] = []
    fail = 0
    schema_empty = 0

    for i, item in enumerate(eval_data, 1):
        inp = item.get("input") or {}
        mr = inp.get("medical_record") if isinstance(inp.get("medical_record"), dict) else None
        ecg = inp.get("ecg_results") if isinstance(inp.get("ecg_results"), list) else []
        sid = str(item.get("id") or f"idx_{i}")
        ids.append(sid)

        user_prompt = build_final_diag_prompt(mr, ecg)

        # Build messages; if YAML says no system role, merge system into user.
        if supports_system:
            messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        else:
            messages = [{"role": "user", "content": f"{sys_prompt}\n\n{user_prompt}"}]

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        ans: Optional[ParsedJSON] = None

        if endpoint == "completions":
            if use_client_tpl:
                prompt = render_template(template, messages)
                payload: Dict[str, Any] = {
                    "model": model_name,
                    "prompt": prompt,
                    token_param: max_tokens,
                    "temperature": temperature,
                }
                if stops:
                    payload["stop"] = stops
                # NOTE: response_format is not reliably supported for /v1/completions in all servers.
                if json_mode:
                    payload["response_format"] = {"type": "json_object"}

                ans = call_completions(
                    s, api_url, headers, payload, timeout_seconds, max_retries, backoff_seconds, errlog
                )

            else:
                # If user isn't providing a template, use chat endpoint with messages
                payload = {"model": model_name, "messages": messages, token_param: max_tokens, "temperature": temperature}
                if stops:
                    payload["stop"] = stops
                if json_mode:
                    payload["response_format"] = {"type": "json_object"}

                ans = call_chat(s, api_url, headers, payload, timeout_seconds, max_retries, backoff_seconds, errlog)

        else:  # chat
            payload = {"model": model_name, "messages": messages, "temperature": temperature, token_param: max_tokens}
            if stops:
                payload["stop"] = stops
            if json_mode:
                payload["response_format"] = {"type": "json_object"}

            ans = call_chat(s, api_url, headers, payload, timeout_seconds, max_retries, backoff_seconds, errlog)

        parsed = coerce_answer(ans)
        rankings.append(parsed["diagnosis_pred_rankings"])
        codes.append(parsed["diagnosis_pred_codes"])

        if ans is None:
            fail += 1
        elif not parsed["diagnosis_pred_rankings"]:
            schema_empty += 1
            append_error_log(errlog, f"结构不匹配或空诊断: id={sid} ans_snippet={str(ans)[:1000]}")

        if total:
            print(f"生成中：{i}/{total}（请求失败{fail}，结构空{schema_empty}）", end="\r", flush=True)

    print(f"\n生成完成：{len(rankings)}/{total}（请求失败{fail}，结构空{schema_empty}）")

    out = [{"id": ids[i], "rankings": rankings[i] if i < len(rankings) else []} for i in range(len(ids))]
    save_json(os.path.join(answers_dir, "final_diagnosis_pred.json"), out)
    print(f"Saved -> {os.path.join(answers_dir, 'final_diagnosis_pred.json')}")


if __name__ == "__main__":
    main()
