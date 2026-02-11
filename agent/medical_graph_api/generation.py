"""生成相关逻辑（Benchmark JSON 输出）。

本项目目标为 benchmark 高分输出，因此主链路仅保留 Benchmark JSON 生成。

注意：为了提升健壮性，Benchmark 模式会对模型输出进行清洗，并尝试从混杂文本中提取合法 JSON。
"""

import asyncio
import json
import logging
import re
from typing import List

from .config import (
    BENCHMARK_TARGET_MAX_DIAGNOSES,
    BENCHMARK_TARGET_MIN_DIAGNOSES,
    ENABLE_LLM_POSTPROCESS,
    ENABLE_PRIMARY_CALIBRATION,
    CHAT_MODEL_NAME,
    FORMAT_MODEL_NAME,
    BENCHMARK_COMMON_DIAGNOSES,
)
from .models import OpenAIChatMessage
from .state import state
from .utils import extract_json_from_text, normalize_benchmark_diagnosis_json

logger = logging.getLogger("MedicalGraphAPI")


def _strip_code_fences(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text.strip()
    s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE | re.MULTILINE)
    s = re.sub(r"```$", "", s, flags=re.MULTILINE)
    return s.strip()


async def _call_llm_text(llm, prompt: str) -> str:
    resp = await asyncio.to_thread(llm.invoke, prompt)
    return getattr(resp, "content", "").strip()


def _render_user_text(messages: List[OpenAIChatMessage]) -> str:
    system_prompt = ""
    user_prompt = ""
    for m in messages:
        if m.role == "system":
            system_prompt += m.content + "\n"
        elif m.role == "user":
            user_prompt += m.content + "\n"
    return system_prompt.strip(), user_prompt.strip()


def _render_target_count_hint() -> str:
    mn = int(BENCHMARK_TARGET_MIN_DIAGNOSES)
    mx = int(BENCHMARK_TARGET_MAX_DIAGNOSES)
    if mn < 1:
        mn = 1
    if mx < mn:
        mx = mn
    return f"严格控制输出条数：请你将诊断数量限制在 {mn} 条，除非特别确定，否则不要输出更多。"


async def _llm_postprocess_names(
    *,
    context: str,
    user_prompt: str,
    current_json: str,
) -> str:
    """通过通用模型对诊断名做二次贴合评测集命名（非硬编码）。

    目标：
    - 同义/别名归一（更贴近评测集常用诊断名）
    - 修正括号/分级/急慢性等修饰（若病例支持）
    - 保持少而准（目标 1-3）
    """

    spec = """
你正在参加心脑血管诊疗 benchmark 评测。你必须只输出一个合法 JSON 对象，且只包含如下结构：
{{
  "诊断": [
    {{"诊断名称": "..."}}
  ]
}}
硬规则：
1) 仅输出 JSON，不要任何解释、不要代码块。
2) 诊断列表最多 10 条，按“最可能”到“次要鉴别”排序。
3) 每条 `诊断名称` 必须是简洁的中文疾病/诊断名称。
""".strip()

    hint = _render_target_count_hint()

    common_diagnoses_str = "、".join(BENCHMARK_COMMON_DIAGNOSES)

    prompt = f"""
{spec}

你将收到：
- 【病例信息】（用户输入）
- 【辅助参考信息】（检索 context）
- 【当前诊断JSON】（可能存在同义词、命名不贴合评测集、分级/括号缺失等问题）

【参考标准名称库】（请优先映射到下列名称，如果确信匹配的话）：
{common_diagnoses_str}

你的任务：
1) 基于病例证据，对【当前诊断JSON】中的诊断名称进行“命名归一与精修”。
2) 优先使用【参考标准名称库】中的名称。如果【当前诊断JSON】中的名称与库中某个名称含义一致（如同义词、包含关系），请替换为库中的标准名称。
3) 处理同义/别名（例如：同一疾病不同表述），选择更常见、更规范、与病例更匹配的诊断名。
4) 若病例明确支持分级/危险分层/急慢性修饰（如“高血压3级(极高危)”“慢性心功能不全急性加重”），则补全；若不确定则不要强行添加。
5) 去除不确定性前缀（考虑/可能/疑似/提示/？/待排等）与无关括号注释。
6) 去重并保持顺序。
7) {hint}

【辅助参考信息】:
{context}

【病例信息】:
{user_prompt}

【当前诊断JSON】:
{current_json}

只输出 JSON：
""".strip()

    content = await _call_llm_text(state.llm_format or state.llm_chat, prompt)
    content = _strip_code_fences(content)
    extracted = extract_json_from_text(content)
    return normalize_benchmark_diagnosis_json(extracted, k=10)


async def _llm_parse_user_prompt(user_prompt: str) -> str:
    """通过通用模型解析用户输入，提取纯净的病例数据，去除 Benchmark 指令。"""
    prompt = f"""
你是一个专业的医学文本处理助手。你将收到一段文本，其中包含“任务指令”、“输出格式要求”以及“病例数据”（如病历文书、ECG结果等）。

你的任务是：
1. 提取并保留所有的【病例数据】。
2. 彻底删除所有的【任务指令】（如“任务：最终诊断...”）、【格式要求】（如“请仅返回 JSON...”）以及其他无关的提示词。
3. 保持病例数据的原始描述和医学术语，不做改写或总结。
4. 返回清洗后的纯净病例描述文本。

【输入文本】：
{user_prompt}

【清洗后的病例数据】：
""".strip()

    return await _call_llm_text(state.llm_format or state.llm_chat, prompt)


async def _llm_primary_calibration(
    *,
    context: str,
    user_prompt: str,
    current_json: str,
) -> str:
    """通过医学模型对 Top1 主诊断做最后校准，避免主诊断偏离。"""

    # 优化：如果只有一个诊断，不存在排序问题，直接跳过
    try:
        obj = json.loads(current_json)
        diags = obj.get("诊断", [])
        if isinstance(diags, list) and len(diags) <= 1:
            return current_json
    except Exception:
        pass

    spec = """
你正在参加心脑血管诊疗 benchmark 评测。你必须只输出一个合法 JSON 对象，且只包含如下结构：
{{
  "诊断": [
    {{"诊断名称": "..."}}
  ]
}}
硬规则：
1) 仅输出 JSON，不要任何解释、不要代码块。
2) 诊断列表最多 10 条，按“最可能”到“次要鉴别”排序。
3) 第 1 条必须是你认为最可能的主诊断。
""".strip()

    hint = _render_target_count_hint()

    prompt = f"""
{spec}

你将收到病例与当前诊断列表。请你：
- 检查当前 Top1 是否为最可能主诊断；如果不是，调整排序并必要时替换为更合适的主诊断。
- 其余诊断保持少而准，去重。
- {hint}

【辅助参考信息】:
{context}

【病例信息】:
{user_prompt}

【当前诊断JSON】:
{current_json}

只输出 JSON：
""".strip()

    content = await _call_llm_text(state.llm_chat, prompt)
    content = _strip_code_fences(content)
    extracted = extract_json_from_text(content)
    return normalize_benchmark_diagnosis_json(extracted, k=10)


async def generate_benchmark_json(context: str, messages: List[OpenAIChatMessage]) -> str:
    system_prompt, user_prompt = _render_user_text(messages)

    benchmark_output_spec = """
你正在参加心脑血管诊疗 benchmark 评测。你必须只输出一个合法 JSON 对象，且只包含如下结构：
{{
  "诊断": [
    {{"诊断名称": "..."}}
  ]
}}
规则：
1) 仅输出 JSON，不要任何解释、不要代码块。
2) 诊断列表最多 5 条，按“最可能”到“次要鉴别”排序。
3) 每条 `诊断名称` 必须是简洁的中文疾病/诊断名称，不要包含“考虑/可能/疑似/提示”等词，不要包含括号注释。
4) 如果信息不足，也必须给出你最可能的诊断（放第1条）。
""".strip()

    target_count_hint = _render_target_count_hint()

    # 阶段1a：医学模型生成候选（纯内生知识，不看 context）
    stage1_medical_prompt = f"""
{system_prompt}

你是一名心脑血管专科医生。请仅基于【用户输入】的病例描述，结合你的专业医学知识，给出“候选诊断清单”。
要求：
1) 必须给出一个最可能、最主要、最急迫的主诊断，并放在第1位。
2) 如果特别确定，也可以额外给出不超过3个候选诊断（包含鉴别诊断也可以），按可能性降序。

【用户输入】:
{user_prompt}

输出候选诊断（文本即可）：
""".strip()

    # 阶段1b：通用模型生成候选（基于 RAG context）
    stage1_general_prompt = f"""
{system_prompt}
你是一名心脑血管专科医生。
请仔细阅读【辅助参考信息】与【用户输入】。
你的任务是根列出该病例可能的诊断候选，不超过2条诊断，可以参考资料里面有用的信息。



【辅助参考信息】:
{context}

【用户输入】:
{user_prompt}

输出候选诊断（文本即可）：
""".strip()

    # 阶段2：通用模型抽取/清洗/初步截断，输出严格 JSON
    stage2_prompt = f"""
{benchmark_output_spec}

下面是医学模型经过深思熟虑后给出的最终候选诊断：
{{candidate}}

请你完成：
- 只抽取“诊断名称”（中文疾病名），不要任何理由。
- 去除不确定性前缀（如：考虑/可能/疑似/提示/？/待排等）与括号补充说明。
- 去重并保留原顺序。
- {target_count_hint}

只输出 JSON：
""".strip()

    logger.info(
        f"⚡ Benchmark 多阶段生成：阶段1a/1b(并行)→阶段1c(多选)→阶段2({FORMAT_MODEL_NAME})"
        + ("→后处理" if ENABLE_LLM_POSTPROCESS else "")
        + ("→主诊断校准" if ENABLE_PRIMARY_CALIBRATION else "")
    )

    try:
        # 并行执行阶段 1a 和 1b
        candidate_medical, candidate_general = await asyncio.gather(
            _call_llm_text(state.llm_chat, stage1_medical_prompt),
            _call_llm_text(state.llm_format or state.llm_chat, stage1_general_prompt),
        )

        # 阶段 1c：多选决策
        stage1_selection_prompt = f"""
{system_prompt}

你是一名心脑血管专家。针对当前病例，我们有两份候选诊断建议：

【建议来源 A】
{candidate_medical}

【建议来源 B】
{candidate_general}

【用户输入】：
{user_prompt}

你的任务：
1. 综合判断上述两份建议，结合病例实际情况，筛选出最准确的诊断清单。
2. 这是一个“多选题”过程，你可以采纳 A 或 B 中的诊断，也可以融合。
3. 输出最终的候选诊断列表，按可能性降序排列。
4. 必须给出一个最确定的主诊断。

输出最终诊断清单（文本即可）：
""".strip()

        candidate_final = await _call_llm_text(state.llm_chat, stage1_selection_prompt)

        # 阶段 2：格式化
        content2 = await _call_llm_text(
            state.llm_format or state.llm_chat,
            stage2_prompt.format(candidate=candidate_final),
        )
        content2 = _strip_code_fences(content2)
        extracted2 = extract_json_from_text(content2)
        normalized = normalize_benchmark_diagnosis_json(extracted2, k=10)

        # 阶段3：通用模型后处理（同义归一/命名贴合集/修饰补全）
        if ENABLE_LLM_POSTPROCESS:
            normalized = await _llm_postprocess_names(
                context=context,
                user_prompt=user_prompt,
                current_json=normalized,
            )

        # 阶段4：医学模型主诊断校准（可选）
        if ENABLE_PRIMARY_CALIBRATION:
            # 只有当诊断数量 > 1 时才进行校准（如果是单诊断，通常已经是确定的，无需调整顺序）
            should_calibrate = True
            try:
                temp_obj = json.loads(normalized)
                temp_diags = temp_obj.get("诊断", [])
                if isinstance(temp_diags, list) and len(temp_diags) <= 1:
                    should_calibrate = False
                    logger.info("⚠️ 仅存在 1 个诊断，跳过主诊断校准(Stage 4)")
            except Exception:
                # 解析失败则依然尝试校准，利用校准过程顺便修复
                should_calibrate = True

            if should_calibrate:
                normalized = await _llm_primary_calibration(
                    context=context,
                    user_prompt=user_prompt,
                    current_json=normalized,
                )

        # 服务端校验：通过则返回，否则做一次最小修复
        try:
            obj = json.loads(normalized)
            diags = obj.get("诊断", []) if isinstance(obj, dict) else []
            if isinstance(diags, list) and 1 <= len(diags) <= 10:
                return normalized
        except Exception:
            pass

        repair_prompt = f"""
{benchmark_output_spec}

下面是你刚才的输出（可能存在格式或内容问题）：
{normalized}

请修复后重新输出，仍然只输出 JSON，且最多 10 条诊断名称：
""".strip()

        repaired = await _call_llm_text(state.llm_format or state.llm_chat, repair_prompt)
        repaired = _strip_code_fences(repaired)
        extracted3 = extract_json_from_text(repaired)
        return normalize_benchmark_diagnosis_json(extracted3, k=10)

    except Exception as e:
        logger.error(f"❌ Benchmark Fail: {e}")
        return json.dumps({"诊断": [{"诊断名称": "服务错误"}]}, ensure_ascii=False)
