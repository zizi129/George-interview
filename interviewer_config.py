from __future__ import annotations

import os

from env_utils import load_env_file

load_env_file()

DEFAULT_INTERVIEWER_ID = "cn_atlas"


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def get_interviewer_options() -> list[dict]:
    return [
        {
            "id": "cn_atlas",
            "display_name": "林知远",
            "subtitle": "中文技术面试",
            "description": "聚焦项目经历与工程实践。",
            "language": "zh",
            "language_label": "中文",
            "ui_notice": "林知远会全程使用中文提问与追问。",
            "avatar_id": _env("AVATAR_ID", "wav2lip_avatar_atlas"),
            "tts_backend": _env("TTS_BACKEND", _env("TTS", "tencent")),
            "tts_ref_file": _env("TTS_REF_FILE", "602004"),
            "tts_ref_text": _env("TTS_REF_TEXT", ""),
        },
        {
            "id": "en_atlas5",
            "display_name": "Alex Morgan",
            "subtitle": "English tech interview",
            "description": "Focused on project depth and engineering decisions.",
            "language": "en",
            "language_label": "English",
            "ui_notice": "Alex Morgan will conduct the interview in English.",
            "avatar_id": _env("ENGLISH_AVATAR_ID", "wav2lip_avatar_atlas5_en_v1"),
            "tts_backend": _env("ENGLISH_TTS_BACKEND", "tencent"),
            "tts_ref_file": _env("ENGLISH_TTS_REF_FILE", "502001"),
            "tts_ref_text": _env("ENGLISH_TTS_REF_TEXT", ""),
        },
    ]


def get_interviewer_option(interviewer_id: str | None) -> dict:
    target_id = (interviewer_id or "").strip() or DEFAULT_INTERVIEWER_ID
    options = get_interviewer_options()
    for option in options:
        if option["id"] == target_id:
            return dict(option)
    for option in options:
        if option["id"] == DEFAULT_INTERVIEWER_ID:
            return dict(option)
    return dict(options[0])
