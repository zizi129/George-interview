from __future__ import annotations

import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from openai import OpenAI

from basereal import BaseReal
from env_utils import load_env_file
from logger import logger

load_env_file()

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_MODEL = "qwen-plus"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_DIMENSION_SCORE = 55
DEFAULT_ZH_INTERVIEWER_NAME = "林知远"
DEFAULT_EN_INTERVIEWER_NAME = "Alex Morgan"
DIMENSION_NAMES = [
    "专业基础",
    "项目深度",
    "问题分析",
    "工程实践",
    "沟通表达",
    "学习成长",
]
SENTENCE_END_CHARS = "。！？!?；;.\n"
SOFT_BREAK_CHARS = "，,：:"

DEFAULT_INTERVIEW_PROMPT = """你是一名名叫 {interviewer_name} 的数字人面试官，正在主持一场实时数字人面试。

当前岗位：{job_title}
当前模式：{interview_mode}
当前输出语言：{dialogue_language}

要求：
1. 全程严格使用 {dialogue_language} 与候选人交流。
2. 开场先用一句话完成自我介绍，例如“我是数字人{interviewer_name}，也是你今天的面试官。”然后再请候选人做简短自我介绍。
3. 之后围绕岗位核心能力、项目经历、技术取舍、工程质量、协作与复盘逐步追问。
4. 一次只问一个主要问题，单次回答控制在 2 到 4 句，避免长篇说教，适合实时 TTS 播放。
5. 如果候选人回答空泛，继续追问职责边界、数据指标、取舍原因、失败经验和验证方式。
6. 不要直接给答案，不要提前做总结，不要在中途评分。
7. 对信息不足保持审慎，允许明确指出“信息还不够，需要继续追问”。
"""

DEFAULT_REPORT_PROMPT = """你是一名严格、公正的技术面试评委。你将基于完整的面试记录输出一份结构化评估。

当前岗位：{job_title}
当前模式：{interview_mode}
当前输出语言：中文

评分规则：
1. 只根据面试记录打分，禁止脑补。
2. 如果证据不足，总分不得高于 60。
3. 90-100 代表显著超出岗位预期；80-89 代表表现扎实；70-79 代表基本合格但有短板；60-69 代表边缘通过；59 及以下代表明显不达标。
4. 没有量化结果、职责边界不清、技术取舍解释不充分时，相关维度不得给高分。
5. 总分必须与六个维度的平均水平基本一致，不得明显高于维度平均分。

输出要求：
1. 只输出 JSON，不要带代码块。
2. JSON 字段必须包含：
   - score: 0-100 整数
   - recommendation: 简短结论，不得出现“建议录用 / 建议进入下一轮 / 建议保留观察 / 暂不推荐 / 不建议录用”等招聘决策措辞，可使用“表现扎实 / 仍需补充证据 / 项目深度较强 / 需要继续提升”等中性结论
   - summary: 80-180 字总结
   - strengths: 2-4 条数组
   - suggestions: 2-4 条数组
   - dimensions: 长度为 6 的数组，每项包含 name、score、comment
3. dimensions 的 name 必须严格使用以下六项：
   - 专业基础
   - 项目深度
   - 问题分析
   - 工程实践
   - 沟通表达
   - 学习成长
4. recommendation、summary、strengths、suggestions、dimensions.comment 必须全部使用中文。无论面试官语言为何，最终评估固定使用中文。
"""

DEFAULT_RESUME_PARSE_PROMPT = """你是一名技术招聘助手，需要先把候选人的简历整理成结构化候选人画像，供技术面试官使用。

规则：
1. 只根据简历内容提取信息，不要脑补。
2. 如果信息缺失，对应字段返回空字符串或空数组。
3. question_seeds 生成 3 到 5 条，必须是面试官可以直接拿来追问的问题方向。
4. risk_flags 生成 0 到 4 条，指出需要在面试中核验的不确定点，例如量化结果缺失、职责边界不清、项目描述偏空泛。
5. resume_summary 控制在 80 到 220 字。
6. 只输出 JSON，不要带代码块。

JSON 字段：
- candidate_name: 字符串
- current_title: 字符串
- years_experience: 字符串
- education_summary: 字符串
- skills: 字符串数组
- project_highlights: 字符串数组
- work_highlights: 字符串数组
- question_seeds: 字符串数组
- risk_flags: 字符串数组
- resume_summary: 字符串
"""

DEFAULT_OUTLINE_PROMPT = """你是一名技术面试设计助手，需要在面试开始前输出一份简洁、可执行、适合单人单次面试的大纲。

规则：
1. 只根据岗位、JD 和候选人简历生成，不要脑补不存在的经历。
2. 大纲既要覆盖岗位核心能力，也要覆盖简历里的关键经历和待核验风险点。
3. 问题要短、直接、可用于实时数字人口播，避免空泛和重复。
4. 面试应适可而止，不要对同一细节无止境穷追不舍。
5. 如果 JD 未提供，禁止出现“JD 提到 / 岗位要求要求 / 当前岗位要求为”之类措辞。
6. 如果简历未提供，禁止出现“你简历里提到 / 结合你的过往项目 / 你之前在某公司”之类措辞，也禁止虚构项目、指标、公司、经历、业务场景。
7. 如果只有岗位名称，则大纲必须保持岗位通用表达，不得写任何个性化背景。
8. 只输出 JSON，不要带代码块。

JSON 字段：
- summary: 40-100 字
- focus_areas: 3-5 条数组，表示本轮主要提问内容
- planned_questions: 4-6 条数组，表示计划中的关键问题
- depth_checks: 3-5 条数组，表示追问时重点核验的深度项
- end_conditions: 3-4 条数组，表示什么时候可以自然结束本轮面试
"""

DEFAULT_PROGRESS_REVIEW_PROMPT = """你是一名面试流程控制助手，不直接向候选人说话，只负责判断这一轮接下来应该继续、收尾还是结束。

决策原则：
1. 如果面试总时长已超过 30 分钟，不要再开启新的问题主题；若只差一个关键缺口，可进入 wrap_up，用最后一个简短问题补齐，否则直接 end。
2. 如果未超过 30 分钟，则根据岗位核心点是否覆盖、简历关键经历是否问到、追问深度是否足够来决定 continue 或 end。
3. 避免对同一细节重复追问；如果职责边界、量化结果、技术取舍、难点处理、复盘方式已经问到多数点，应考虑转入结束。
4. 用户轮次明显不足时，不要过早结束。
5. 只输出 JSON，不要带代码块。

JSON 字段：
- decision: continue | wrap_up | end
- reason: 20-80 字，说明原因
- progress_summary: 40-120 字，概括当前进展
- covered_points: 0-5 条数组，已覆盖重点
- remaining_points: 0-5 条数组，仍建议覆盖的重点
- focus_prompt: 0-1 条字符串，指出下一步唯一最值得关注的重点；若 decision=end，可写为空字符串
"""


def _load_prompt(filename: str, fallback: str) -> str:
    prompt_path = PROMPTS_DIR / filename
    if prompt_path.is_file():
        return prompt_path.read_text(encoding="utf-8").strip()
    return fallback.strip()


def _get_client() -> OpenAI:
    api_key = (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("LLM_API_KEY")
    )
    if not api_key:
        raise RuntimeError("未找到 DASHSCOPE_API_KEY / OPENAI_API_KEY / LLM_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
    )


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _looks_like_english_title(text: str) -> bool:
    cleaned = _clean_text(text)
    return bool(cleaned) and not re.search(r"[\u4e00-\u9fff]", cleaned) and bool(re.search(r"[A-Za-z]", cleaned))


@lru_cache(maxsize=128)
def translate_job_title(job_title: str, target_language: str = "zh") -> str:
    cleaned = _clean_text(job_title)
    if not cleaned:
        return ""
    if target_language != "en" or _looks_like_english_title(cleaned):
        return cleaned

    try:
        started_at = time.perf_counter()
        client = _get_client()
        completion = client.chat.completions.create(
            model=os.getenv("JOB_TITLE_TRANSLATION_MODEL", os.getenv("LLM_MODEL", DEFAULT_MODEL)),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate the provided job title into concise, natural English for an interview context. "
                        "Return only the translated job title. Do not add quotes, labels, explanations, or extra punctuation."
                    ),
                },
                {"role": "user", "content": cleaned},
            ],
        )
        translated = _clean_text(completion.choices[0].message.content if completion.choices else "")
        translated = translated.strip("`\"' ")
        if not translated or len(translated) > 120:
            return cleaned
        logger.info("job title translated in %ss: %s -> %s", time.perf_counter() - started_at, cleaned, translated)
        return translated
    except Exception:
        logger.exception("translate_job_title")
        return cleaned


def _split_ready_sentences(buffer: str) -> tuple[list[str], str]:
    ready: list[str] = []
    start = 0

    for index, char in enumerate(buffer):
        if char in SENTENCE_END_CHARS:
            sentence = _clean_text(buffer[start : index + 1])
            if sentence:
                ready.append(sentence)
            start = index + 1
        elif char in SOFT_BREAK_CHARS and (index - start) >= 24:
            sentence = _clean_text(buffer[start : index + 1])
            if sentence:
                ready.append(sentence)
            start = index + 1

    return ready, buffer[start:]


def _merge_history_messages(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for item in history:
        role = "assistant" if item.get("role") == "assistant" else "user"
        content = _clean_text(str(item.get("content", "")))
        if not content:
            continue
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n" + content
        else:
            merged.append({"role": role, "content": content})
    return merged


def _build_chat_messages(nerfreal: BaseReal) -> list[dict[str, str]]:
    context = nerfreal.get_interview_context()
    interviewer_language = "en" if context.get("interviewer_language") == "en" else "zh"
    interview_mode = (
        ("voice interview" if context["interview_mode"] == "voice" else "text interview")
        if interviewer_language == "en"
        else ("语音面试" if context["interview_mode"] == "voice" else "文字面试")
    )
    dialogue_language = "English" if interviewer_language == "en" else "中文"
    interviewer_name = (
        _clean_text(str(context.get("interviewer_name", "")))
        or (DEFAULT_EN_INTERVIEWER_NAME if interviewer_language == "en" else DEFAULT_ZH_INTERVIEWER_NAME)
    )
    system_prompt = _load_prompt(
        "digital_interviewer.txt",
        DEFAULT_INTERVIEW_PROMPT,
    ).format(
        job_title=context["job_title"] or "未指定岗位",
        interview_mode=interview_mode,
        interviewer_name=interviewer_name,
        dialogue_language=dialogue_language,
    )
    job_requirements_prompt = _build_job_requirements_prompt_suffix(context)
    if job_requirements_prompt:
        system_prompt += "\n\n" + job_requirements_prompt
    resume_prompt = _build_resume_prompt_suffix(context)
    if resume_prompt:
        system_prompt += "\n\n" + resume_prompt
    management_prompt = _build_interview_management_prompt_suffix(context)
    if management_prompt:
        system_prompt += "\n\n" + management_prompt

    messages = [{"role": "system", "content": system_prompt}]
    history = _merge_history_messages(nerfreal.get_recent_history(limit=24))
    messages.extend(history)
    return messages


def build_interview_opening(nerfreal: BaseReal) -> str:
    context = nerfreal.get_interview_context()
    job_title = _clean_text(str(context.get("job_title", "") or "")) or "当前岗位"
    interviewer_language = "en" if context.get("interviewer_language") == "en" else "zh"
    interviewer_name = (
        _clean_text(str(context.get("interviewer_name", "")))
        or (DEFAULT_EN_INTERVIEWER_NAME if interviewer_language == "en" else DEFAULT_ZH_INTERVIEWER_NAME)
    )
    if interviewer_language == "en":
        return f"Hello, I'm {interviewer_name}, the digital interviewer for today. Welcome to the {job_title} interview. Please start with a brief self-introduction."
    return f"你好，我是数字人{interviewer_name}，也是你今天的面试官。欢迎参加{job_title}面试，请先做一个简短的自我介绍。"


def _emit_assistant_text(nerfreal: BaseReal, text: str, generation_id: int) -> None:
    clean_text = _clean_text(text)
    if not clean_text or not nerfreal.is_generation_current(generation_id):
        return
    nerfreal.add_interview_message("assistant", clean_text, {"source": "llm"})
    nerfreal.put_msg_txt(clean_text, {"source": "llm"})


def llm_response(message: str, nerfreal: BaseReal) -> None:
    generation_id = nerfreal.begin_generation()
    start = time.perf_counter()

    try:
        progress = review_interview_progress(nerfreal)
        nerfreal.set_interview_context(outline_progress=progress)
        logger.info(
            "interview decision: sessionid=%s decision=%s reason=%s",
            nerfreal.sessionid,
            progress["decision"],
            progress["reason"],
        )

        if progress["decision"] == "end":
            closing = build_interview_closing(nerfreal, progress.get("reason", ""))
            if nerfreal.is_generation_current(generation_id):
                nerfreal.add_interview_message("assistant", closing, {"source": "closing"})
                nerfreal.put_msg_txt(closing, {"source": "closing"})
                nerfreal.mark_interview_finished(
                    progress.get("reason", ""),
                    time.time() + _estimate_speech_duration_seconds(closing),
                )
            return

        if progress["decision"] == "wrap_up":
            nerfreal.increment_wrap_up_turns()

        client = _get_client()
        init_done = time.perf_counter()
        logger.info("llm Time init: %ss", init_done - start)

        completion = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
            messages=_build_chat_messages(nerfreal),
            stream=True,
            stream_options={"include_usage": True},
        )

        buffer = ""
        first_chunk = True

        for chunk in completion:
            if not nerfreal.is_generation_current(generation_id):
                logger.info("llm generation cancelled: sessionid=%s", nerfreal.sessionid)
                break
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue

            if first_chunk:
                logger.info("llm Time to first chunk: %ss", time.perf_counter() - start)
                first_chunk = False

            buffer += delta
            ready_sentences, buffer = _split_ready_sentences(buffer)
            for sentence in ready_sentences:
                _emit_assistant_text(nerfreal, sentence, generation_id)

        if nerfreal.is_generation_current(generation_id):
            _emit_assistant_text(nerfreal, buffer, generation_id)
            logger.info("llm Time to last chunk: %ss", time.perf_counter() - start)
    except Exception:
        logger.exception("llm_response")
        if nerfreal.is_generation_current(generation_id):
            interviewer_language = "en" if nerfreal.get_interview_context().get("interviewer_language") == "en" else "zh"
            fallback = (
                "The interviewer is temporarily unavailable. Please try again in a moment, or continue with another question."
                if interviewer_language == "en"
                else "当前面试官连接异常，请稍后重试，或换个问题继续。"
            )
            nerfreal.add_interview_message("assistant", fallback, {"source": "llm_error"})
            nerfreal.put_msg_txt(fallback, {"source": "llm_error"})


def _extract_json_payload(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        return {}

    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    try:
        data = json.loads(cleaned[start : end + 1])
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _build_resume_prompt_suffix(context: dict[str, Any]) -> str:
    profile = context.get("resume_profile") or {}
    if not isinstance(profile, dict) or not any(profile.values()):
        return ""

    interviewer_language = "en" if context.get("interviewer_language") == "en" else "zh"
    sections = [
        (
            "Candidate resume information (treat the following only as claims from the resume and verify them during the interview, not as proven facts):"
            if interviewer_language == "en"
            else "候选人简历信息（以下内容仅视为候选人简历陈述，需要在面试中核验，不要直接当作已证实事实）："
        )
    ]
    candidate_name = _clean_text(str(profile.get("candidate_name", "")))
    current_title = _clean_text(str(profile.get("current_title", "")))
    years_experience = _clean_text(str(profile.get("years_experience", "")))
    education_summary = _clean_text(str(profile.get("education_summary", "")))
    resume_summary = _clean_text(str(profile.get("resume_summary", "")))
    skills = _ensure_text_list(profile.get("skills"), [])[:8]
    project_highlights = _ensure_text_list(profile.get("project_highlights"), [])[:4]
    work_highlights = _ensure_text_list(profile.get("work_highlights"), [])[:4]
    question_seeds = _ensure_text_list(profile.get("question_seeds"), [])[:5]
    risk_flags = _ensure_text_list(profile.get("risk_flags"), [])[:4]

    if candidate_name:
        sections.append(f"{'Candidate name' if interviewer_language == 'en' else '候选人姓名'}：{candidate_name}")
    if current_title:
        sections.append(f"{'Current or latest title' if interviewer_language == 'en' else '当前/最近岗位'}：{current_title}")
    if years_experience:
        sections.append(f"{'Years of experience' if interviewer_language == 'en' else '工作年限'}：{years_experience}")
    if education_summary:
        sections.append(f"{'Education' if interviewer_language == 'en' else '教育背景'}：{education_summary}")
    if resume_summary:
        sections.append(f"{'Resume summary' if interviewer_language == 'en' else '简历摘要'}：{resume_summary}")
    if skills:
        sections.append(("Core skills: " if interviewer_language == 'en' else "核心技能：") + ("; ".join(skills) if interviewer_language == 'en' else "、".join(skills)))
    if project_highlights:
        sections.append(("Project highlights:\n- " if interviewer_language == 'en' else "项目亮点：\n- ") + "\n- ".join(project_highlights))
    if work_highlights:
        sections.append(("Work experience highlights:\n- " if interviewer_language == 'en' else "工作经历重点：\n- ") + "\n- ".join(work_highlights))
    if question_seeds:
        sections.append(("Suggested follow-up directions:\n- " if interviewer_language == 'en' else "建议优先追问：\n- ") + "\n- ".join(question_seeds))
    if risk_flags:
        sections.append(("Items to verify:\n- " if interviewer_language == 'en' else "待核验点：\n- ") + "\n- ".join(risk_flags))

    sections.append(
        (
            "Additional instruction: prioritize the experiences most relevant to the target role, and keep probing for ownership boundaries, quantified outcomes, technical trade-offs, difficult cases, failures, and retrospectives."
            if interviewer_language == "en"
            else "提问要求补充：优先围绕与目标岗位最相关的经历展开，继续追问职责边界、量化结果、技术取舍、难点处理、失败经验和复盘方式。"
        )
    )
    return "\n".join(sections)


def _build_job_requirements_prompt_suffix(context: dict[str, Any]) -> str:
    raw_requirements = str(context.get("job_requirements", "") or "").strip()
    if not raw_requirements:
        return ""

    interviewer_language = "en" if context.get("interviewer_language") == "en" else "zh"
    if interviewer_language == "en":
        return (
            "Role requirements / JD (if provided, treat this as a high-priority basis for organizing questions and follow-ups. "
            "Focus on the responsibilities, required skills, business context, and evaluation priorities below, then compare the candidate's experience against them):\n"
            + raw_requirements
        )

    return (
        "岗位要求 / JD（如果提供，优先依据以下内容组织提问与追问。重点围绕岗位职责、必备技能、业务场景和评估重点展开，再结合候选人的简历和回答判断匹配度）：\n"
        + raw_requirements
    )


def _normalize_text_items(value: Any, limit: int, fallback: list[str] | None = None) -> list[str]:
    items: list[str] = []
    if isinstance(value, list):
        for item in value:
            text = _clean_text(str(item))
            if text:
                items.append(text)
            if len(items) >= limit:
                break
    if items:
        return items
    return list((fallback or [])[:limit])


def _has_resume_context(context: dict[str, Any]) -> bool:
    profile = context.get("resume_profile") or {}
    if not isinstance(profile, dict):
        return False
    for value in profile.values():
        if isinstance(value, list) and any(_clean_text(str(item)) for item in value):
            return True
        if isinstance(value, dict) and value:
            return True
        if not isinstance(value, (list, dict)) and _clean_text(str(value)):
            return True
    return False


def _has_job_requirements_context(context: dict[str, Any]) -> bool:
    return bool(_clean_text(str(context.get("job_requirements", "") or "")))


def _extract_text_points(raw_text: str, limit: int = 4) -> list[str]:
    text = str(raw_text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    chunks = re.split(r"[\n；;。]", text)
    points: list[str] = []
    for chunk in chunks:
        cleaned = _clean_text(re.sub(r"^[\-\*\d\.\)\s]+", "", chunk))
        cleaned = re.sub(r"^(岗位职责|任职要求|职位描述|职责|要求)[:：]?", "", cleaned).strip()
        if not cleaned:
            continue
        if cleaned in points:
            continue
        points.append(cleaned[:48])
        if len(points) >= limit:
            break
    return points


def _get_role_template(job_title: str) -> dict[str, list[str]]:
    normalized = _clean_text(job_title).lower()
    if any(keyword in normalized for keyword in ("产品", "product", "pm")):
        return {
            "focus_areas": [
                "用户问题判断与需求优先级",
                "方案设计、范围控制与取舍依据",
                "数据指标定义、验证与迭代",
                "跨团队协作与推动落地",
                "复盘与持续优化能力",
            ],
            "planned_questions": [
                "请讲一个你主导或深度参与的产品功能，重点说明问题判断、方案选择和上线结果。",
                "当需求价值、开发成本和上线时间冲突时，你通常如何做优先级决策？",
                "你如何定义一个功能是否真的有效？请举例说明关键指标和验证方式。",
                "请讲一次你和研发、设计或运营意见不一致的经历，你是如何推动达成共识的？",
                "如果一个功能上线后效果不及预期，你通常如何定位问题并决定下一步动作？",
            ],
            "depth_checks": [
                "是否能说清用户问题、约束条件和取舍理由",
                "是否给出清晰的数据指标与验证方法",
                "是否说明自己的实际角色与推动动作",
                "是否具备复盘与迭代思路",
            ],
        }
    if any(keyword in normalized for keyword in ("算法", "machine learning", "ml", "ai", "模型", "数据科学")):
        return {
            "focus_areas": [
                "核心项目职责边界与实际贡献",
                "模型或方案选型的取舍依据",
                "训练、评估、上线与效果验证",
                "问题定位、优化与复盘能力",
                "工程化与协作落地能力",
            ],
            "planned_questions": [
                "请讲一个你负责最深的项目，重点说明你的职责边界、关键动作和最终结果。",
                "当时做过最重要的一次技术取舍是什么？你为什么这么选？",
                "你如何定义和验证效果提升？请说明指标、实验或对照方式。",
                "遇到效果不稳定或线上问题时，你通常如何排查和修正？",
                "请讲一个你在工程落地或跨团队协作中真正推动过的改进点。",
            ],
            "depth_checks": [
                "是否能说明职责边界和关键贡献",
                "是否给出量化结果或验证证据",
                "是否说清技术取舍和排查路径",
                "是否兼顾工程落地与协作细节",
            ],
        }
    if any(keyword in normalized for keyword in ("前端", "后端", "开发", "engineer", "工程师", "客户端", "测试")):
        return {
            "focus_areas": [
                "核心项目职责与系统理解",
                "技术方案设计与关键取舍",
                "工程质量、稳定性与性能意识",
                "问题定位、联调与复盘能力",
                "协作效率与交付意识",
            ],
            "planned_questions": [
                "请讲一个你负责最深的项目，重点说明你负责的模块、难点和落地结果。",
                "当时最关键的一次技术方案取舍是什么？你为什么这么决定？",
                "你如何保证代码质量、稳定性或性能？请举一个具体例子。",
                "遇到线上故障或复杂 bug 时，你通常怎么定位和推进解决？",
                "请讲一个你和上下游协作推进交付的经历，重点说明你的作用。",
            ],
            "depth_checks": [
                "是否能讲清模块边界和设计取舍",
                "是否具备质量、稳定性或性能意识",
                "是否说明排障路径与验证方式",
                "是否体现协作与交付能力",
            ],
        }
    return {
        "focus_areas": [
            "岗位核心能力与相关经历",
            "关键任务的判断与执行方式",
            "结果验证、问题处理与复盘",
            "协作沟通与推进能力",
        ],
        "planned_questions": [
            f"请讲一个和{job_title or '当前岗位'}最相关的经历，重点说明你的职责、做法和结果。",
            "当遇到目标、资源或时间冲突时，你通常如何做判断和取舍？",
            "你如何判断一项工作是否做得好？请说明你的验证方式。",
            "请讲一个你遇到困难并最终解决的问题，重点说明过程和复盘。",
        ],
        "depth_checks": [
            "是否能讲清职责边界和实际动作",
            "是否能给出结果证据或验证方式",
            "是否具备问题分析与复盘能力",
        ],
    }


def _build_outline_source_text(context: dict[str, Any]) -> str:
    has_jd = _has_job_requirements_context(context)
    has_resume = _has_resume_context(context)
    sections = [
        f"目标岗位：{_clean_text(str(context.get('job_title', '') or '未指定岗位')) or '未指定岗位'}",
        f"岗位要求 / JD：{('已提供' if has_jd else '未提供')}",
        f"候选人简历：{('已提供' if has_resume else '未提供')}",
        "硬性约束：只允许使用明确提供的信息；缺失的信息必须保持空白，不得补写、联想或虚构。",
    ]
    job_requirements = str(context.get("job_requirements", "") or "").strip()
    if has_jd:
        sections.append("岗位要求 / JD 内容：\n" + job_requirements[:3000])
    else:
        sections.append("岗位要求 / JD 内容：未提供")

    profile = context.get("resume_profile") or {}
    if has_resume and isinstance(profile, dict) and any(profile.values()):
        candidate_name = _clean_text(str(profile.get("candidate_name", "")))
        current_title = _clean_text(str(profile.get("current_title", "")))
        years_experience = _clean_text(str(profile.get("years_experience", "")))
        education_summary = _clean_text(str(profile.get("education_summary", "")))
        resume_summary = _clean_text(str(profile.get("resume_summary", "")))
        skills = _normalize_text_items(profile.get("skills"), 8)
        project_highlights = _normalize_text_items(profile.get("project_highlights"), 4)
        work_highlights = _normalize_text_items(profile.get("work_highlights"), 4)
        question_seeds = _normalize_text_items(profile.get("question_seeds"), 5)
        risk_flags = _normalize_text_items(profile.get("risk_flags"), 4)

        if candidate_name:
            sections.append(f"候选人姓名：{candidate_name}")
        if current_title:
            sections.append(f"当前/最近岗位：{current_title}")
        if years_experience:
            sections.append(f"工作年限：{years_experience}")
        if education_summary:
            sections.append(f"教育背景：{education_summary}")
        if resume_summary:
            sections.append(f"简历摘要：{resume_summary}")
        if skills:
            sections.append("核心技能：" + "、".join(skills))
        if project_highlights:
            sections.append("项目亮点：\n- " + "\n- ".join(project_highlights))
        if work_highlights:
            sections.append("工作经历重点：\n- " + "\n- ".join(work_highlights))
        if question_seeds:
            sections.append("简历建议追问：\n- " + "\n- ".join(question_seeds))
        if risk_flags:
            sections.append("简历待核验点：\n- " + "\n- ".join(risk_flags))
    else:
        sections.append("候选人简历内容：未提供")

    return "\n\n".join(section for section in sections if section)


def _fallback_interview_outline(context: dict[str, Any]) -> dict[str, Any]:
    job_title = _clean_text(str(context.get("job_title", ""))) or "当前岗位"
    role_template = _get_role_template(job_title)
    profile = context.get("resume_profile") or {}
    has_resume = _has_resume_context(context)
    has_jd = _has_job_requirements_context(context)
    requirement_points = _extract_text_points(context.get("job_requirements", ""), 3)
    question_seeds = _normalize_text_items(profile.get("question_seeds") if isinstance(profile, dict) else [], 5)
    risk_flags = _normalize_text_items(profile.get("risk_flags") if isinstance(profile, dict) else [], 4)
    project_highlights = _normalize_text_items(profile.get("project_highlights") if isinstance(profile, dict) else [], 3)
    resume_summary = _clean_text(str(profile.get("resume_summary", ""))) if isinstance(profile, dict) else ""

    focus_areas = list(role_template["focus_areas"])
    if requirement_points:
        focus_areas.extend([f"岗位要求中的重点：{item}" for item in requirement_points[:2]])
    if has_resume and risk_flags:
        focus_areas.append("简历中的待核验风险点")
    focus_areas = _normalize_text_items(focus_areas, 5)

    planned_questions = question_seeds[:]
    for point in requirement_points:
        planned_questions.append(f"请结合具体经历说明你在“{point}”上的实际做法、判断依据和结果。")
        if len(planned_questions) >= 6:
            break
    if not planned_questions:
        planned_questions = list(role_template["planned_questions"])
    if project_highlights and len(planned_questions) < 6:
        planned_questions.append(f"简历里提到“{project_highlights[0]}”，请具体展开职责边界、难点和结果。")
    planned_questions = _normalize_text_items(planned_questions, 6)

    depth_checks = list(role_template["depth_checks"])
    if risk_flags:
        depth_checks.extend([f"核验点：{item}" for item in risk_flags[:2]])
    if requirement_points:
        depth_checks.extend([f"是否覆盖岗位要求中的“{item}”" for item in requirement_points[:2]])
    depth_checks = _normalize_text_items(depth_checks, 5)

    summary_parts = [f"围绕{job_title}岗位核心能力展开，优先覆盖关键经历、判断依据和结果验证。"]
    if has_jd:
        summary_parts.append("已对齐已填写的岗位要求。")
    else:
        summary_parts.append("当前未填写 JD，将按岗位通用能力组织问题。")
    if has_resume and resume_summary:
        summary_parts.append("已结合简历中的关键信息安排追问。")
    elif not has_resume:
        summary_parts.append("当前未上传简历，将从候选人的现场回答中逐步建立判断。")

    return {
        "summary": _clean_text(" ".join(summary_parts))[:120],
        "focus_areas": focus_areas,
        "planned_questions": planned_questions,
        "depth_checks": depth_checks,
        "end_conditions": [
            "岗位核心问题已覆盖",
            "职责边界、结果、取舍与复盘已有足够信息",
            "继续追问的新增收益已经有限",
            "若时长接近或超过30分钟，应自然收尾",
        ],
    }


def _normalize_interview_outline(raw: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    fallback = _fallback_interview_outline(context)
    outline = raw if isinstance(raw, dict) else {}
    return {
        "summary": _clean_text(str(outline.get("summary", "")))[:120] or fallback["summary"],
        "focus_areas": _normalize_text_items(outline.get("focus_areas"), 5, fallback["focus_areas"]),
        "planned_questions": _normalize_text_items(outline.get("planned_questions"), 6, fallback["planned_questions"]),
        "depth_checks": _normalize_text_items(outline.get("depth_checks"), 5, fallback["depth_checks"]),
        "end_conditions": _normalize_text_items(outline.get("end_conditions"), 4, fallback["end_conditions"]),
    }


def build_initial_outline_progress(outline: dict[str, Any]) -> dict[str, Any]:
    focus_areas = _normalize_text_items((outline or {}).get("focus_areas"), 5)
    return {
        "decision": "continue",
        "reason": "面试刚开始，先按大纲完成自我介绍与核心经历摸底。",
        "progress_summary": "已生成面试大纲，接下来先从候选人最相关的经历切入，再逐步核验关键能力。",
        "covered_points": [],
        "remaining_points": focus_areas,
        "focus_prompt": focus_areas[0] if focus_areas else "先从候选人最相关的经历切入。",
    }


def _create_json_completion(model_name: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    client = _get_client()
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    content = response.choices[0].message.content or ""
    return _extract_json_payload(content)


def generate_interview_outline(nerfreal: BaseReal) -> dict[str, Any]:
    context = nerfreal.get_interview_context()
    started_at = time.perf_counter()
    outline = _fallback_interview_outline(context)
    enable_llm_outline = str(os.getenv("ENABLE_LLM_INTERVIEW_OUTLINE", "") or "").strip().lower() in {"1", "true", "yes", "on"}
    if not enable_llm_outline:
        logger.info("interview outline built in %ss: sessionid=%s source=deterministic", time.perf_counter() - started_at, nerfreal.sessionid)
        return outline

    if not (_has_job_requirements_context(context) or _has_resume_context(context)):
        logger.info("interview outline built in %ss: sessionid=%s source=deterministic_no_profile", time.perf_counter() - started_at, nerfreal.sessionid)
        return outline

    prompt = DEFAULT_OUTLINE_PROMPT
    try:
        payload = _create_json_completion(
            os.getenv("INTERVIEW_OUTLINE_MODEL", os.getenv("LLM_MODEL", DEFAULT_MODEL)),
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": "请基于以下面试上下文生成结构化面试大纲：\n\n" + _build_outline_source_text(context),
                },
            ],
        )
        outline = _normalize_interview_outline(payload, context)
        logger.info("interview outline built in %ss: sessionid=%s source=llm", time.perf_counter() - started_at, nerfreal.sessionid)
        return outline
    except Exception:
        logger.exception("generate_interview_outline")
        return outline


def format_interview_outline_for_log(outline: dict[str, Any], context: dict[str, Any]) -> str:
    normalized = _normalize_interview_outline(outline, context)
    lines = [
        "================ 面试大纲 ================",
        f"sessionid: {context.get('sessionid', '-') or '-'}",
        f"岗位: {_clean_text(str(context.get('job_title', '') or '未指定岗位')) or '未指定岗位'}",
        f"模式: {'语音面试' if context.get('interview_mode') == 'voice' else '文字面试'}",
        f"概述: {normalized['summary']}",
        "主要提问内容:",
    ]
    lines.extend([f"{index}. {item}" for index, item in enumerate(normalized["focus_areas"], start=1)])
    lines.append("计划问题:")
    lines.extend([f"{index}. {item}" for index, item in enumerate(normalized["planned_questions"], start=1)])
    lines.append("深挖检查项:")
    lines.extend([f"- {item}" for item in normalized["depth_checks"]])
    lines.append("自然结束条件:")
    lines.extend([f"- {item}" for item in normalized["end_conditions"]])
    lines.append("=========================================")
    return "\n".join(lines)


def _fallback_outline_progress(
    context: dict[str, Any],
    outline: dict[str, Any],
    user_turns: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    previous = context.get("outline_progress") or {}
    covered_points = _normalize_text_items(previous.get("covered_points"), 5)
    remaining_points = _normalize_text_items(previous.get("remaining_points"), 5, outline.get("focus_areas", []))
    wrap_up_turns = int(context.get("wrap_up_turns") or 0)

    if elapsed_seconds >= 1800 and wrap_up_turns >= 1:
        return {
            "decision": "end",
            "reason": "面试已超过30分钟，且已经给过最后一次收尾追问，应当自然结束。",
            "progress_summary": "当前已进入收尾完成阶段，不再继续开启新的问题方向。",
            "covered_points": covered_points,
            "remaining_points": remaining_points[:1],
            "focus_prompt": "",
        }
    if elapsed_seconds >= 1800:
        return {
            "decision": "wrap_up",
            "reason": "面试已超过30分钟，需要尽快自然收尾。",
            "progress_summary": "如果还差一个关键缺口，只补最后一个短问题；否则直接结束。",
            "covered_points": covered_points,
            "remaining_points": remaining_points[:1],
            "focus_prompt": remaining_points[0] if remaining_points else "",
        }
    if user_turns < 3:
        return {
            "decision": "continue",
            "reason": "当前候选人回答轮次较少，仍需继续完成基础信息采集。",
            "progress_summary": "优先完成核心经历摸底与岗位相关度核验。",
            "covered_points": covered_points,
            "remaining_points": remaining_points,
            "focus_prompt": remaining_points[0] if remaining_points else "",
        }
    if user_turns >= 5 and len(remaining_points) <= 1:
        return {
            "decision": "end",
            "reason": "主要问题已基本覆盖，继续追问的收益有限，可以自然结束。",
            "progress_summary": "当前信息已足以支撑本轮复盘，可进入结束流程。",
            "covered_points": covered_points,
            "remaining_points": remaining_points,
            "focus_prompt": "",
        }
    return {
        "decision": "continue",
        "reason": "仍有关键点未完全覆盖，继续围绕剩余重点推进。",
        "progress_summary": "继续按照大纲推进，并优先补齐剩余重点。",
        "covered_points": covered_points,
        "remaining_points": remaining_points,
        "focus_prompt": remaining_points[0] if remaining_points else "",
    }


def _normalize_outline_progress(
    raw: dict[str, Any],
    context: dict[str, Any],
    outline: dict[str, Any],
    user_turns: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    fallback = _fallback_outline_progress(context, outline, user_turns, elapsed_seconds)
    progress = raw if isinstance(raw, dict) else {}
    decision = _clean_text(str(progress.get("decision", ""))).lower()
    if decision not in {"continue", "wrap_up", "end"}:
        decision = fallback["decision"]

    wrap_up_turns = int(context.get("wrap_up_turns") or 0)
    if elapsed_seconds >= 1800:
        if wrap_up_turns >= 1:
            decision = "end"
        elif decision == "continue":
            decision = "wrap_up"
    elif decision == "wrap_up":
        decision = "continue"

    if user_turns < 3 and decision == "end" and elapsed_seconds < 1800:
        decision = "continue"

    covered_points = _normalize_text_items(progress.get("covered_points"), 5, fallback["covered_points"])
    remaining_points = _normalize_text_items(progress.get("remaining_points"), 5, fallback["remaining_points"])
    focus_prompt = _clean_text(str(progress.get("focus_prompt", "")))[:120]
    if decision == "end":
        focus_prompt = ""
    elif not focus_prompt:
        focus_prompt = fallback["focus_prompt"]

    return {
        "decision": decision,
        "reason": _clean_text(str(progress.get("reason", "")))[:120] or fallback["reason"],
        "progress_summary": _clean_text(str(progress.get("progress_summary", "")))[:180] or fallback["progress_summary"],
        "covered_points": covered_points,
        "remaining_points": remaining_points,
        "focus_prompt": focus_prompt,
    }


def _format_recent_transcript(nerfreal: BaseReal, limit: int = 24) -> str:
    history = nerfreal.get_recent_history(limit=limit)
    if not history:
        return "暂无对话"
    lines = []
    for item in history:
        speaker = "面试官" if item.get("role") == "assistant" else "候选人"
        lines.append(f"{speaker}: {_clean_text(str(item.get('content', '')))}")
    return "\n".join(lines)


def review_interview_progress(nerfreal: BaseReal) -> dict[str, Any]:
    context = nerfreal.get_interview_context()
    outline = _normalize_interview_outline(context.get("interview_outline") or {}, context)
    user_turns = nerfreal.get_user_turn_count()
    elapsed_seconds = nerfreal.get_interview_elapsed_seconds()

    if user_turns <= 0:
        return build_initial_outline_progress(outline)
    if elapsed_seconds >= 1800 and int(context.get("wrap_up_turns") or 0) >= 1:
        return _fallback_outline_progress(context, outline, user_turns, elapsed_seconds)

    prompt = DEFAULT_PROGRESS_REVIEW_PROMPT
    previous_progress = context.get("outline_progress") or build_initial_outline_progress(outline)
    review_payload = "\n\n".join(
        [
            f"面试总时长（秒）：{int(elapsed_seconds)}",
            f"候选人回答轮次：{user_turns}",
            f"已进入收尾追问次数：{int(context.get('wrap_up_turns') or 0)}",
            "当前面试大纲：",
            "概述：" + outline["summary"],
            "主要提问内容：\n- " + "\n- ".join(outline["focus_areas"]),
            "计划问题：\n- " + "\n- ".join(outline["planned_questions"]),
            "深挖检查项：\n- " + "\n- ".join(outline["depth_checks"]),
            "自然结束条件：\n- " + "\n- ".join(outline["end_conditions"]),
            "上一轮进度判断：",
            "decision: " + _clean_text(str(previous_progress.get("decision", ""))),
            "reason: " + _clean_text(str(previous_progress.get("reason", ""))),
            "progress_summary: " + _clean_text(str(previous_progress.get("progress_summary", ""))),
            "covered_points:\n- " + ("\n- ".join(_normalize_text_items(previous_progress.get("covered_points"), 5)) or "无"),
            "remaining_points:\n- " + ("\n- ".join(_normalize_text_items(previous_progress.get("remaining_points"), 5)) or "无"),
            "最近对话记录：",
            _format_recent_transcript(nerfreal),
        ]
    )

    try:
        started_at = time.perf_counter()
        payload = _create_json_completion(
            os.getenv("INTERVIEW_REVIEW_MODEL", os.getenv("LLM_MODEL", DEFAULT_MODEL)),
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": review_payload},
            ],
        )
        progress = _normalize_outline_progress(payload, context, outline, user_turns, elapsed_seconds)
        logger.info(
            "interview progress reviewed in %ss: sessionid=%s decision=%s",
            time.perf_counter() - started_at,
            nerfreal.sessionid,
            progress["decision"],
        )
        return progress
    except Exception:
        logger.exception("review_interview_progress")
        return _fallback_outline_progress(context, outline, user_turns, elapsed_seconds)


def _build_interview_management_prompt_suffix(context: dict[str, Any]) -> str:
    outline = context.get("interview_outline") or {}
    progress = context.get("outline_progress") or {}
    elapsed_seconds = float(context.get("interview_elapsed_seconds") or 0.0)
    focus_areas = _normalize_text_items(outline.get("focus_areas"), 5)
    planned_questions = _normalize_text_items(outline.get("planned_questions"), 6)
    depth_checks = _normalize_text_items(outline.get("depth_checks"), 5)
    end_conditions = _normalize_text_items(outline.get("end_conditions"), 4)
    covered_points = _normalize_text_items(progress.get("covered_points"), 5)
    remaining_points = _normalize_text_items(progress.get("remaining_points"), 5)
    decision = _clean_text(str(progress.get("decision", ""))).lower() or "continue"
    focus_prompt = _clean_text(str(progress.get("focus_prompt", "")))

    lines = [
        "面试推进控制：",
        "1. 始终以当前面试大纲为准推进，优先覆盖仍未完成的重点。",
        "2. 不要在同一细节上无止境追问；当职责边界、量化结果、技术取舍、难点处理与复盘已基本问清时，应及时收束。",
        "3. 问题保持短、直接、自然，一次只推进一个主要点。",
    ]
    if outline.get("summary"):
        lines.append("大纲概述：" + _clean_text(str(outline.get("summary", ""))))
    if focus_areas:
        lines.append("主要提问内容：\n- " + "\n- ".join(focus_areas))
    if planned_questions:
        lines.append("关键问题参考：\n- " + "\n- ".join(planned_questions))
    if depth_checks:
        lines.append("深挖检查项：\n- " + "\n- ".join(depth_checks))
    if covered_points:
        lines.append("已覆盖重点：\n- " + "\n- ".join(covered_points))
    if remaining_points:
        lines.append("待覆盖重点：\n- " + "\n- ".join(remaining_points))
    if end_conditions:
        lines.append("自然结束条件：\n- " + "\n- ".join(end_conditions))
    if focus_prompt:
        lines.append("下一步唯一重点：" + focus_prompt)
    if decision == "wrap_up" or elapsed_seconds >= 1800:
        lines.append("当前已进入收尾阶段：不要开启新的主题；如果还有一个关键缺口，只允许再问一个简短问题，随后自然结束。")
    else:
        lines.append("如果主要重点已覆盖且继续追问收益有限，可以自然结束，不必为了追问而追问。")
    return "\n".join(lines)


def build_interview_closing(nerfreal: BaseReal, reason: str = "") -> str:
    interviewer_language = "en" if nerfreal.get_interview_context().get("interviewer_language") == "en" else "zh"
    if interviewer_language == "en":
        return "Alright, that will be the end of this round. Thank you for your answers. I will generate your feedback report shortly."
    if "30" in reason or "超" in reason:
        return "好的，今天这轮面试我先了解到这里。感谢你的回答，稍后我会为你生成本轮复盘报告。"
    return "好的，本轮面试就先到这里。感谢你的回答，稍后我会为你生成本轮复盘报告。"


def _estimate_speech_duration_seconds(text: str) -> float:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return 2.5
    return max(2.5, min(12.0, len(compact) / 5.0 + 1.2))


def _clamp_score(value: Any, default: int = DEFAULT_DIMENSION_SCORE) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = default
    return max(0, min(100, score))


def _ensure_text_list(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        items = [_clean_text(str(item)) for item in value if _clean_text(str(item))]
        if items:
            return items[:4]
    return fallback


def _fallback_resume_profile(resume_text: str, job_title: str) -> dict[str, Any]:
    lines = [line.strip() for line in (resume_text or "").splitlines() if line.strip()]
    summary_source = _clean_text(" ".join(lines[:10]) or resume_text)[:220]
    first_line = lines[0] if lines else ""
    candidate_name = first_line if first_line and len(first_line) <= 20 and "简历" not in first_line else ""
    education_summary = next(
        (
            line for line in lines
            if any(keyword in line for keyword in ("大学", "学院", "本科", "硕士", "博士", "专科"))
        ),
        "",
    )
    target_job = job_title or "目标岗位"
    return {
        "candidate_name": candidate_name,
        "current_title": "",
        "years_experience": "",
        "education_summary": education_summary,
        "skills": [],
        "project_highlights": lines[:3],
        "work_highlights": [],
        "question_seeds": [
            f"请挑一段和{target_job}最相关的经历，说明你的具体职责。",
            "请展开讲一个你参与最深的项目，重点说技术取舍和落地结果。",
            "如果简历里有一段你最想被重点考察的经历，请先详细介绍。",
        ],
        "risk_flags": ["当前为自动提炼结果，建议在面试中核验关键经历和量化结果。"],
        "resume_summary": summary_source or "已上传简历，可在面试中继续核验具体经历。",
    }


def _normalize_resume_profile(raw: dict[str, Any], resume_text: str, job_title: str) -> dict[str, Any]:
    fallback = _fallback_resume_profile(resume_text, job_title)
    profile = raw if isinstance(raw, dict) else {}
    resume_summary = _clean_text(str(profile.get("resume_summary", ""))) or fallback["resume_summary"]
    return {
        "candidate_name": _clean_text(str(profile.get("candidate_name", ""))),
        "current_title": _clean_text(str(profile.get("current_title", ""))),
        "years_experience": _clean_text(str(profile.get("years_experience", ""))),
        "education_summary": _clean_text(str(profile.get("education_summary", ""))) or fallback["education_summary"],
        "skills": _ensure_text_list(profile.get("skills"), fallback["skills"])[:8],
        "project_highlights": _ensure_text_list(profile.get("project_highlights"), fallback["project_highlights"])[:4],
        "work_highlights": _ensure_text_list(profile.get("work_highlights"), fallback["work_highlights"])[:4],
        "question_seeds": _ensure_text_list(profile.get("question_seeds"), fallback["question_seeds"])[:5],
        "risk_flags": _ensure_text_list(profile.get("risk_flags"), fallback["risk_flags"])[:4],
        "resume_summary": resume_summary[:220],
    }


def extract_resume_profile(resume_text: str, job_title: str = "") -> dict[str, Any]:
    cleaned_text = (resume_text or "").strip()
    if not cleaned_text:
        return _fallback_resume_profile("", job_title)

    prompt = _load_prompt("resume_parser.txt", DEFAULT_RESUME_PARSE_PROMPT)
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=os.getenv("RESUME_PARSE_MODEL", os.getenv("LLM_MODEL", DEFAULT_MODEL)),
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "目标岗位："
                        + (job_title or "未指定")
                        + "\n\n请把下面这份候选人简历整理成结构化 JSON：\n\n"
                        + cleaned_text[:12000]
                    ),
                },
            ],
        )
        content = response.choices[0].message.content or ""
        payload = _extract_json_payload(content)
        if not payload:
            return _fallback_resume_profile(cleaned_text, job_title)
        return _normalize_resume_profile(payload, cleaned_text, job_title)
    except Exception:
        logger.exception("extract_resume_profile")
        return _fallback_resume_profile(cleaned_text, job_title)


def _normalize_dimensions(raw_dimensions: Any) -> list[dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    if isinstance(raw_dimensions, list):
        for item in raw_dimensions:
            if not isinstance(item, dict):
                continue
            name = _clean_text(str(item.get("name", "")))
            if name in DIMENSION_NAMES:
                mapped[name] = item

    dimensions: list[dict[str, Any]] = []
    for name in DIMENSION_NAMES:
        item = mapped.get(name, {})
        dimensions.append(
            {
                "name": name,
                "score": _clamp_score(item.get("score"), DEFAULT_DIMENSION_SCORE),
                "comment": _clean_text(str(item.get("comment", "")))
                or "证据有限，按保守标准评分。",
            }
        )
    return dimensions


def _fallback_report(nerfreal: BaseReal, reason: str) -> dict[str, Any]:
    transcript = nerfreal.get_interview_transcript_text()
    user_turns = nerfreal.get_user_turn_count()
    limited = user_turns < 3 or len(transcript) < 180
    base_score = 58 if limited else 64
    dimensions = [
        {
            "name": name,
            "score": base_score if limited else base_score + 2,
            "comment": "可用面试信息有限，采用保守评分。",
        }
        for name in DIMENSION_NAMES
    ]
    return {
        "score": base_score,
        "recommendation": "需补充更多证据",
        "summary": "本轮面试记录不足以支撑高置信度结论，建议补充更多岗位相关追问后再做进一步判断。",
        "strengths": ["能够进入基本问答流程", "已形成一轮可供复核的面试记录"],
        "suggestions": ["增加岗位核心问题和项目细节追问", "补充量化结果、技术取舍和复盘细节"],
        "dimensions": dimensions,
        "meta": {
            "job_title": nerfreal.get_interview_context()["job_title"],
            "reason": reason,
        },
    }


def _normalize_report(raw: dict[str, Any], nerfreal: BaseReal) -> dict[str, Any]:
    dimensions = _normalize_dimensions(raw.get("dimensions"))
    average_score = round(sum(item["score"] for item in dimensions) / len(dimensions))
    overall_score = _clamp_score(raw.get("score"), average_score)
    overall_score = min(overall_score, average_score + 3)

    transcript = nerfreal.get_interview_transcript_text()
    user_turns = nerfreal.get_user_turn_count()
    if user_turns < 3 or len(transcript) < 180:
        overall_score = min(overall_score, 60)
        for item in dimensions:
            item["score"] = min(item["score"], 65)

    recommendation = _clean_text(str(raw.get("recommendation", ""))) or "需补充更多证据"
    summary = _clean_text(str(raw.get("summary", ""))) or "本轮面试已完成，建议结合更多追问信息再做进一步判断。"
    strengths = _ensure_text_list(
        raw.get("strengths"),
        ["具备基础沟通能力", "能够完成当前轮次问答"],
    )
    suggestions = _ensure_text_list(
        raw.get("suggestions"),
        ["补充更具体的项目细节", "增加量化结果与技术取舍说明"],
    )

    return {
        "score": overall_score,
        "recommendation": recommendation,
        "summary": summary,
        "strengths": strengths,
        "suggestions": suggestions,
        "dimensions": dimensions,
        "meta": {
            "job_title": nerfreal.get_interview_context()["job_title"],
            "user_turns": user_turns,
            "transcript_chars": len(transcript),
        },
    }


def generate_interview_report(nerfreal: BaseReal) -> dict[str, Any]:
    transcript = nerfreal.get_interview_transcript_text()
    if not transcript.strip():
        return _fallback_report(nerfreal, "empty_transcript")

    context = nerfreal.get_interview_context()
    interview_mode = "语音面试" if context["interview_mode"] == "voice" else "文字面试"
    report_language = "中文"
    prompt = _load_prompt("interview_report.txt", DEFAULT_REPORT_PROMPT).format(
        job_title=context["job_title"] or "未指定岗位",
        interview_mode=interview_mode,
        report_language=report_language,
    )

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=os.getenv("INTERVIEW_REPORT_MODEL", os.getenv("LLM_MODEL", DEFAULT_MODEL)),
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": "请基于以下面试记录输出严格的 JSON 评估结果：\n\n" + transcript,
                },
            ],
        )
        content = response.choices[0].message.content or ""
        payload = _extract_json_payload(content)
        if not payload:
            return _fallback_report(nerfreal, "invalid_json")
        return _normalize_report(payload, nerfreal)
    except Exception:
        logger.exception("generate_interview_report")
        return _fallback_report(nerfreal, "report_generation_error")