"""通用工具函数集合。

当前主要包含：
- `extract_json_from_text`：从模型输出的混杂文本中尽可能提取出一个合法 JSON 对象

该工具用于增强 Benchmark 模式下的鲁棒性，避免模型输出夹带解释文字/代码块导致解析失败。
"""

import json
import re
from typing import Any, Dict, List


def extract_json_from_text(text: str) -> str:
    """从文本中提取第一个合法 JSON 对象（字符串形式）。

    策略：
    1. 先尝试整体直接 `json.loads`
    2. 若失败，截取第一个 `{` 到最后一个 `}` 之间的片段再尝试解析
    3. 最终失败则返回一个最小可用的兜底 JSON，防止调用方崩溃

    Args:
        text: 任意文本，可能是模型输出（可能包含 Markdown 代码块等杂质）。

    Returns:
        str: 可被 `json.loads` 解析的 JSON 字符串。
    """
    text = text.strip()

    # 1) 直接解析
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # 2) 尝试截取第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    # 3) 兜底：返回最小合法 JSON，避免上层报错
    return json.dumps({"诊断": [{"诊断名称": "生成结果解析失败"}]}, ensure_ascii=False)


def _normalize_cn_diagnosis_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip()
    if not s:
        return ""

    # 去掉代码块/引号/多余空白
    s = s.replace("```", " ")
    s = re.sub(r"\s+", " ", s)

    # 去掉常见口癖/不确定性前缀
    s = re.sub(r"^(考虑|疑似|可能|拟诊|诊断为|初步诊断为|诊断：|诊断:|提示|倾向于)\s*", "", s)

    # 去掉括号内补充说明（保留核心名词）
    s = re.sub(r"[（(].*?[)）]", "", s)

    # 去掉末尾标点
    s = s.strip(" 	\r\n，,。.;；:：")

    return s


def normalize_benchmark_diagnosis_json(raw_json_str: str, k: int = 10) -> str:
    """将模型输出的JSON规范化为评测友好的最小结构。

    评测关注：前K个 `诊断名称` 的排序与命中，且Precision@K会惩罚噪声。
    因此这里做：
    - 只保留 `诊断` 列表
    - 只保留每项的 `诊断名称`
    - 清洗名称、去重、裁剪到K
    - 必要时兜底返回一个最小合法JSON
    """
    try:
        obj: Dict[str, Any] = json.loads(raw_json_str)
    except Exception:
        return json.dumps({"诊断": [{"诊断名称": "生成结果解析失败"}]}, ensure_ascii=False)

    diags: List[Dict[str, Any]] = []

    if isinstance(obj, dict) and isinstance(obj.get("诊断"), list):
        for item in obj.get("诊断", []):
            if isinstance(item, dict):
                name = _normalize_cn_diagnosis_name(item.get("诊断名称", ""))
            else:
                name = _normalize_cn_diagnosis_name(str(item))
            if name:
                diags.append({"诊断名称": name})
    else:
        # 兼容模型直接输出字符串列表/单字符串
        if isinstance(obj, list):
            for x in obj:
                name = _normalize_cn_diagnosis_name(str(x))
                if name:
                    diags.append({"诊断名称": name})
        elif isinstance(obj, dict):
            # 常见替代字段
            cand = obj.get("diagnosis") or obj.get("diagnoses") or obj.get("disease")
            if isinstance(cand, list):
                for x in cand:
                    name = _normalize_cn_diagnosis_name(str(x))
                    if name:
                        diags.append({"诊断名称": name})

    # 去重（保持顺序）
    seen = set()
    deduped: List[Dict[str, str]] = []
    for d in diags:
        n = d["诊断名称"]
        if n not in seen:
            seen.add(n)
            deduped.append(d)

    if not deduped:
        deduped = [{"诊断名称": "生成失败"}]

    return json.dumps({"诊断": deduped[:k]}, ensure_ascii=False)
