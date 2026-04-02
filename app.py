###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

# server.py
from flask import Flask, render_template,send_from_directory,request, jsonify
from flask_sockets import Sockets
import base64
import json
#import gevent
#from gevent import pywsgi
#from geventwebsocket.handler import WebSocketHandler
import re
import numpy as np
from threading import Thread,Event
#import multiprocessing
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription,RTCIceServer,RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender
from webrtc import HumanPlayer
from basereal import BaseReal

import argparse
import random
import shutil
import asyncio
import time
import torch
import os
import hmac
import hashlib
import subprocess
import tempfile
from typing import Dict
from threading import Lock, Timer
from datetime import datetime
from pathlib import Path
from copy import deepcopy
from logger import logger
import gc
from env_utils import load_env_file
from user_db import (
    init_db,
    deduct_quota as db_deduct_quota,
    save_user_resume,
    get_user_resume,
    save_interview_report,
    get_user_interview_reports,
)
from auth import setup_auth_routes, get_user_from_request
from payment import setup_payment_routes
from interviewer_config import DEFAULT_INTERVIEWER_ID, get_interviewer_option, get_interviewer_options
from llm import (
    build_initial_outline_progress,
    build_interview_opening,
    extract_resume_profile,
    format_interview_outline_for_log,
    generate_interview_outline,
    generate_interview_report,
    llm_response,
    translate_job_title,
)
from interview_agent import interview_agent_process_turn
from resume_utils import extract_resume_content
load_env_file()
init_db()

app = Flask(__name__)
#sockets = Sockets(app)
nerfreals:Dict[int, BaseReal] = {} #sessionid:BaseReal
opt = None
model = None
avatar = None
avatar_bundle_cache: Dict[tuple[str, str], object] = {}
avatar_bundle_cache_lock = Lock()

TENCENT_ASR_HOST = "asr.tencentcloudapi.com"
TENCENT_ASR_ACTION = "SentenceRecognition"
TENCENT_ASR_VERSION = "2019-06-14"
TENCENT_ASR_SERVICE = "asr"
MAX_RESUME_FILE_BYTES = 5 * 1024 * 1024

def ok_json(data: dict | None = None, status: int = 200):
    payload = {"code": 0}
    if data:
        payload.update(data)
    return web.json_response(payload, status=status)


def error_json(message: str, status: int = 400):
    return web.json_response({"code": -1, "msg": message}, status=status)


def get_session_real(sessionid: int) -> BaseReal:
    nerfreal = nerfreals.get(sessionid)
    if nerfreal is None:
        raise KeyError(f"无效 sessionid: {sessionid}")
    return nerfreal


def build_invalid_session_state(reason: str = "invalid_session") -> dict:
    interviewer = get_interviewer_option(DEFAULT_INTERVIEWER_ID)
    return {
        "messages": [],
        "context": {
            "job_title": "",
            "job_requirements": "",
            "interview_mode": "text",
            "resume_file_name": "",
            "resume_profile": {},
            "interviewer_id": interviewer["id"],
            "interviewer_name": interviewer["display_name"],
            "interviewer_language": interviewer["language"],
            "interviewer_language_label": interviewer["language_label"],
            "interview_outline": {
                "summary": "",
                "focus_areas": [],
                "planned_questions": [],
                "depth_checks": [],
                "end_conditions": [],
            },
            "outline_progress": {
                "decision": "continue",
                "reason": "",
                "progress_summary": "",
                "covered_points": [],
                "remaining_points": [],
                "focus_prompt": "",
            },
            "interview_started_at": 0.0,
            "interview_elapsed_seconds": 0.0,
            "interview_finished": False,
            "interview_finish_reason": "",
            "interview_report_ready_at": 0.0,
            "wrap_up_turns": 0,
        },
        "speaking": False,
        "sessionid": 0,
        "invalid_session": True,
        "reason": reason,
    }


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _sanitize_string_list(value, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        text = str(item or "").strip()
        if text:
            items.append(text[:160])
        if len(items) >= limit:
            break
    return items


def _sanitize_basic_info_block(raw) -> dict:
    if not isinstance(raw, dict):
        raw = {}
    return {
        "phone": str(raw.get("phone", "") or "").strip()[:40],
        "email": str(raw.get("email", "") or "").strip()[:120],
        "gender": str(raw.get("gender", "") or "").strip()[:20],
        "birth_date": str(raw.get("birth_date", "") or "").strip()[:40],
        "hometown": str(raw.get("hometown", "") or "").strip()[:80],
        "residence": str(raw.get("residence", "") or "").strip()[:80],
        "desired_position": str(raw.get("desired_position", "") or "").strip()[:120],
        "desired_location": str(raw.get("desired_location", "") or "").strip()[:120],
    }


def _sanitize_education_block_list(value) -> list[dict]:
    out: list[dict] = []
    if not isinstance(value, list):
        return out
    for item in value[:6]:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "school": str(item.get("school", "") or "").strip()[:120],
                "degree": str(item.get("degree", "") or "").strip()[:60],
                "major": str(item.get("major", "") or "").strip()[:120],
                "start_date": str(item.get("start_date", "") or "").strip()[:40],
                "end_date": str(item.get("end_date", "") or "").strip()[:40],
                "detail": str(item.get("detail", "") or "").strip()[:800],
            }
        )
    return out


def _sanitize_work_experience_list(value) -> list[dict]:
    out: list[dict] = []
    if not isinstance(value, list):
        return out
    for item in value[:8]:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "company": str(item.get("company", "") or "").strip()[:120],
                "title": str(item.get("title", "") or item.get("position", "") or "").strip()[:120],
                "type": str(item.get("type", "") or "").strip()[:40],
                "start_date": str(item.get("start_date", "") or "").strip()[:40],
                "end_date": str(item.get("end_date", "") or "").strip()[:40],
                "description": str(item.get("description", "") or "").strip()[:1200],
            }
        )
    return out


def _sanitize_resume_profile(payload) -> dict:
    if not isinstance(payload, dict):
        return {}

    return {
        "candidate_name": str(payload.get("candidate_name", "") or "").strip()[:60],
        "current_title": str(payload.get("current_title", "") or "").strip()[:120],
        "years_experience": str(payload.get("years_experience", "") or "").strip()[:60],
        "education_summary": str(payload.get("education_summary", "") or "").strip()[:160],
        "basic_info": _sanitize_basic_info_block(payload.get("basic_info")),
        "education": _sanitize_education_block_list(payload.get("education")),
        "work_experience": _sanitize_work_experience_list(payload.get("work_experience")),
        "skills": _sanitize_string_list(payload.get("skills"), 8),
        "project_highlights": _sanitize_string_list(payload.get("project_highlights"), 4),
        "work_highlights": _sanitize_string_list(payload.get("work_highlights"), 4),
        "question_seeds": _sanitize_string_list(payload.get("question_seeds"), 5),
        "risk_flags": _sanitize_string_list(payload.get("risk_flags"), 4),
        "resume_summary": str(payload.get("resume_summary", "") or "").strip()[:240],
    }


def _sanitize_text_block(value, limit: int = 4000) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:limit]


def apply_env_overrides(parsed_opt):
    parsed_opt.avatar_id = os.getenv("AVATAR_ID", parsed_opt.avatar_id)
    parsed_opt.tts = os.getenv("TTS_BACKEND", os.getenv("TTS", parsed_opt.tts))
    parsed_opt.REF_FILE = os.getenv("TTS_REF_FILE", parsed_opt.REF_FILE)
    parsed_opt.REF_TEXT = os.getenv("TTS_REF_TEXT", parsed_opt.REF_TEXT)
    parsed_opt.TTS_SERVER = os.getenv("TTS_SERVER", parsed_opt.TTS_SERVER)
    parsed_opt.model = os.getenv("HUMAN_MODEL", parsed_opt.model)
    parsed_opt.transport = os.getenv("TRANSPORT", parsed_opt.transport)
    parsed_opt.push_url = os.getenv("PUSH_URL", parsed_opt.push_url)
    parsed_opt.listenport = _env_int("LISTEN_PORT", parsed_opt.listenport)
    parsed_opt.max_session = _env_int("MAX_SESSION", parsed_opt.max_session)
    return parsed_opt


def _serialize_interviewer_option(option: dict) -> dict:
    preview_url = f"/interviewer/preview/{option['id']}"
    return {
        "id": option["id"],
        "display_name": option["display_name"],
        "subtitle": option["subtitle"],
        "description": option["description"],
        "language": option["language"],
        "language_label": option["language_label"],
        "ui_notice": option["ui_notice"],
        "preview_url": preview_url,
    }


def _find_avatar_preview_path(avatar_id: str) -> Path | None:
    preview_dir = Path("./data/avatars") / avatar_id / "full_imgs"
    if not preview_dir.is_dir():
        return None

    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        matches = sorted(preview_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _get_avatar_bundle(avatar_id: str):
    cache_key = (opt.model, avatar_id)
    cached_bundle = avatar_bundle_cache.get(cache_key)
    if cached_bundle is not None:
        return cached_bundle

    with avatar_bundle_cache_lock:
        cached_bundle = avatar_bundle_cache.get(cache_key)
        if cached_bundle is not None:
            return cached_bundle

        if opt.model == 'wav2lip':
            from lipreal import load_avatar as load_avatar_bundle
        elif opt.model == 'musetalk':
            from musereal import load_avatar as load_avatar_bundle
        elif opt.model == 'ultralight':
            from lightreal import load_avatar as load_avatar_bundle
        else:
            raise RuntimeError(f"unsupported model for avatar loading: {opt.model}")

        avatar_bundle = load_avatar_bundle(avatar_id)
        avatar_bundle_cache[cache_key] = avatar_bundle
        return avatar_bundle


async def interviewer_prepare(request):
    try:
        params = await request.json()
        interviewer_id = str(params.get("interviewer_id", "") or "").strip() or DEFAULT_INTERVIEWER_ID
        interviewer = get_interviewer_option(interviewer_id)
        avatar_id = interviewer["avatar_id"]
        await asyncio.get_event_loop().run_in_executor(None, _get_avatar_bundle, avatar_id)
        return ok_json({"data": {"interviewer_id": interviewer["id"], "avatar_id": avatar_id}})
    except Exception as e:
        logger.exception("interviewer_prepare")
        return error_json(str(e), status=500)


def _sha256_hex(payload: bytes | str) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hmac_sha256(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _get_tencent_asr_config() -> dict:
    secret_id = _env_str("TENCENT_ASR_SECRET_ID", _env_str("TENCENT_SECRET_ID"))
    secret_key = _env_str("TENCENT_ASR_SECRET_KEY", _env_str("TENCENT_SECRET_KEY"))
    if not secret_id or not secret_key:
        raise RuntimeError("缺少腾讯云 ASR 凭证，请先配置 TENCENT_SECRET_ID 和 TENCENT_SECRET_KEY。")

    return {
        "secret_id": secret_id,
        "secret_key": secret_key,
        "region": _env_str("TENCENT_ASR_REGION", "ap-shanghai"),
        "engine_model_type": _env_str("TENCENT_ASR_ENGINE_MODEL_TYPE", "16k_zh"),
        "hotword_list": _env_str("TENCENT_ASR_HOTWORD_LIST"),
    }


def _build_tencent_asr_headers(secret_id: str, secret_key: str, region: str, body_bytes: bytes) -> dict:
    timestamp = int(datetime.utcnow().timestamp())
    date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")
    canonical_headers = (
        "content-type:application/json; charset=utf-8\n"
        f"host:{TENCENT_ASR_HOST}\n"
    )
    signed_headers = "content-type;host"
    canonical_request = "\n".join([
        "POST",
        "/",
        "",
        canonical_headers,
        signed_headers,
        _sha256_hex(body_bytes),
    ])

    credential_scope = f"{date}/{TENCENT_ASR_SERVICE}/tc3_request"
    string_to_sign = "\n".join([
        "TC3-HMAC-SHA256",
        str(timestamp),
        credential_scope,
        _sha256_hex(canonical_request),
    ])

    secret_date = _hmac_sha256(("TC3" + secret_key).encode("utf-8"), date)
    secret_service = _hmac_sha256(secret_date, TENCENT_ASR_SERVICE)
    secret_signing = _hmac_sha256(secret_service, "tc3_request")
    signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    authorization = (
        "TC3-HMAC-SHA256 "
        f"Credential={secret_id}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )

    return {
        "Authorization": authorization,
        "Content-Type": "application/json; charset=utf-8",
        "Host": TENCENT_ASR_HOST,
        "X-TC-Action": TENCENT_ASR_ACTION,
        "X-TC-Version": TENCENT_ASR_VERSION,
        "X-TC-Timestamp": str(timestamp),
        "X-TC-Region": region,
    }


def _convert_audio_to_wav_bytes(filebytes: bytes, filename: str) -> bytes:
    if not filebytes:
        raise RuntimeError("上传的语音文件为空。")

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("当前环境缺少 ffmpeg，无法处理录音格式。")

    suffix = Path(filename or "voice.webm").suffix or ".webm"
    with tempfile.TemporaryDirectory(prefix="livetalking-asr-") as temp_dir:
        input_path = Path(temp_dir) / f"input{suffix}"
        output_path = Path(temp_dir) / "output.wav"
        input_path.write_bytes(filebytes)

        completed = subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-i",
                str(input_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                str(output_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0 or not output_path.exists():
            logger.error("ffmpeg convert failed: %s", completed.stderr.decode("utf-8", errors="ignore"))
            raise RuntimeError("语音格式转换失败，请重新录音后再试。")

        wav_bytes = output_path.read_bytes()
        if len(wav_bytes) > 3 * 1024 * 1024:
            raise RuntimeError("录音文件过大，请控制在 60 秒以内后重试。")
        return wav_bytes


def _normalize_tencent_asr_error(code: str, message: str) -> str:
    mapping = {
        "FailedOperation.UserNotRegistered": "腾讯云语音识别服务未开通，请先在腾讯云控制台开通 ASR。",
        "FailedOperation.UserHasNoAmount": "腾讯云 ASR 资源不足，请检查余额或资源包。",
        "FailedOperation.UserHasNoFreeAmount": "腾讯云 ASR 资源不足，请检查余额或资源包。",
        "AuthFailure.SignatureFailure": "腾讯云 ASR 鉴权失败，请检查 SecretId / SecretKey。",
        "AuthFailure.SecretIdNotFound": "腾讯云 ASR 鉴权失败，请检查 SecretId / SecretKey。",
        "AuthFailure.TokenFailure": "腾讯云 ASR 鉴权失败，请检查 SecretId / SecretKey。",
        "InvalidParameterValue.ErrorInvalidVoiceFormat": "上传的录音格式腾讯云不支持。",
        "InvalidParameterValue.ErrorVoicedataTooLong": "录音过长，请控制在 60 秒以内。",
        "InvalidParameter.ErrorContentlength": "录音数据长度无效，请重新录音后重试。",
        "FailedOperation.ErrorRecognize": "腾讯云 ASR 未识别到有效语音，请重试。",
    }
    detail = mapping.get(code, message or code or "腾讯云 ASR 调用失败。")
    return f"{detail} ({code})" if code else detail


async def _tencent_sentence_recognition(wav_bytes: bytes) -> str:
    config = _get_tencent_asr_config()
    payload = {
        "ProjectId": 0,
        "SubServiceType": 2,
        "EngSerViceType": config["engine_model_type"],
        "SourceType": 1,
        "VoiceFormat": "wav",
        "Data": base64.b64encode(wav_bytes).decode("utf-8"),
        "DataLen": len(wav_bytes),
        "WordInfo": 0,
        "FilterDirty": 0,
        "FilterModal": 0,
        "FilterPunc": 0,
        "ConvertNumMode": 1,
    }
    if config["hotword_list"]:
        payload["HotwordList"] = config["hotword_list"]

    body_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    headers = _build_tencent_asr_headers(
        config["secret_id"],
        config["secret_key"],
        config["region"],
        body_bytes,
    )

    timeout = aiohttp.ClientTimeout(total=45)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"https://{TENCENT_ASR_HOST}", data=body_bytes, headers=headers) as response:
            response_text = await response.text()

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("腾讯云 ASR 返回格式异常，请稍后重试。") from exc

    result = payload.get("Response", {})
    error = result.get("Error")
    if error:
        raise RuntimeError(_normalize_tencent_asr_error(error.get("Code", ""), error.get("Message", "")))

    text = (result.get("Result") or "").strip()
    if not text:
        raise RuntimeError("未识别到有效语音，请重试。")
    return text


#####webrtc###############################
pcs = set()

def randN(N)->int:
    '''生成长度为 N的随机数 '''
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)

def build_nerfreal(sessionid:int, interviewer_id: str | None = None)->BaseReal:
    session_opt = deepcopy(opt)
    interviewer = get_interviewer_option(interviewer_id)
    session_opt.sessionid = sessionid
    session_opt.avatar_id = interviewer["avatar_id"]
    session_opt.tts = interviewer["tts_backend"]
    session_opt.REF_FILE = interviewer["tts_ref_file"]
    session_opt.REF_TEXT = interviewer["tts_ref_text"] or None
    session_opt.interviewer_id = interviewer["id"]
    session_opt.interviewer_language = interviewer["language"]
    avatar_bundle = _get_avatar_bundle(session_opt.avatar_id)

    if session_opt.model == 'wav2lip':
        from lipreal import LipReal
        nerfreal = LipReal(session_opt,model,avatar_bundle)
    elif session_opt.model == 'musetalk':
        from musereal import MuseReal
        nerfreal = MuseReal(session_opt,model,avatar_bundle)
    # elif opt.model == 'ernerf':
    #     from nerfreal import NeRFReal
    #     nerfreal = NeRFReal(opt,model,avatar)
    elif session_opt.model == 'ultralight':
        from lightreal import LightReal
        nerfreal = LightReal(session_opt,model,avatar_bundle)
    return nerfreal

#@app.route('/offer', methods=['POST'])
async def offer(request):
    user = get_user_from_request(request)
    if not user:
        return web.json_response({"code": -1, "msg": "请先登录"}, status=401)

    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    interviewer_id = str(params.get("interviewer_id", "") or "").strip() or DEFAULT_INTERVIEWER_ID

    # if len(nerfreals) >= opt.max_session:
    #     logger.info('reach max session')
    #     return web.Response(
    #         content_type="application/json",
    #         text=json.dumps(
    #             {"code": -1, "msg": "reach max session"}
    #         ),
    #     )
    sessionid = randN(6) #len(nerfreals)
    nerfreals[sessionid] = None
    logger.info('sessionid=%d, session num=%d',sessionid,len(nerfreals))
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid, interviewer_id)
    nerfreals[sessionid] = nerfreal
    
    #ice_server = RTCIceServer(urls='stun:stun.l.google.com:19302')
    ice_server = RTCIceServer(urls='stun:stun.freeswitch.org:3478')
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[ice_server]))
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            nerfreals.pop(sessionid, None)
        if pc.connectionState == "closed":
            pcs.discard(pc)
            nerfreals.pop(sessionid, None)
            # gc.collect()

    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)
    capabilities = RTCRtpSender.getCapabilities("video")
    preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
    transceiver = pc.getTransceivers()[1]
    transceiver.setCodecPreferences(preferences)

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    #return jsonify({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "sessionid":sessionid,
                "interviewer_id": interviewer_id,
            }
        ),
    )

async def human(request):
    try:
        user = get_user_from_request(request)
        if not user:
            return web.json_response({"code": -1, "msg": "请先登录"}, status=401)

        params = await request.json()

        sessionid = int(params.get('sessionid',0))
        nerfreal = get_session_real(sessionid)
        text = str(params.get('text', '')).strip()
        input_type = params.get('type', 'chat')

        if params.get('interrupt'):
            nerfreal.flush_talk()

        if not text:
            return error_json("text 不能为空")

        if input_type == 'echo':
            nerfreal.add_interview_message('assistant', text, {'source': 'echo'})
            nerfreal.put_msg_txt(text)
        elif input_type == 'chat':
            nerfreal.add_interview_message(
                'user',
                text,
                {'source': params.get('source', 'text'), 'mode': params.get('mode', 'chat')},
            )
            asyncio.get_event_loop().run_in_executor(None, interview_agent_process_turn, text, nerfreal)
        else:
            return error_json(f"unsupported human type: {input_type}")

        return ok_json({"msg":"ok"})
    except Exception as e:
        logger.exception('exception:')
        return error_json(str(e), status=500)

async def interrupt_talk(request):
    try:
        params = await request.json()

        sessionid = int(params.get('sessionid',0))
        get_session_real(sessionid).flush_talk()
        return ok_json({"msg":"ok"})
    except Exception as e:
        logger.exception('exception:')
        return error_json(str(e), status=500)

async def humanaudio(request):
    try:
        form= await request.post()
        sessionid = int(form.get('sessionid',0))
        fileobj = form["file"]
        filename=fileobj.filename
        filebytes=fileobj.file.read()
        get_session_real(sessionid).put_audio_file(filebytes)
        return ok_json({"msg":"ok"})
    except Exception as e:
        logger.exception('exception:')
        return error_json(str(e), status=500)


async def resume_upload(request):
    try:
        form = await request.post()
        fileobj = form.get("file")
        if fileobj is None:
            return error_json("缺少简历文件。", status=400)

        filename = getattr(fileobj, "filename", "") or "resume.txt"
        filebytes = fileobj.file.read()
        if not filebytes:
            return error_json("上传的简历文件为空。", status=400)
        if len(filebytes) > MAX_RESUME_FILE_BYTES:
            return error_json("简历文件过大，请控制在 5MB 以内。", status=400)

        job_title = str(form.get("job_title", "") or "").strip()
        resume_content = await asyncio.to_thread(extract_resume_content, filebytes, filename)
        resume_text = str(resume_content.get("text", "") or "")
        profile = await asyncio.get_event_loop().run_in_executor(None, extract_resume_profile, resume_text, job_title)
        return ok_json(
            {
                "data": {
                    "file_name": filename,
                    "ocr_used": bool(resume_content.get("ocr_used")),
                    "profile": _sanitize_resume_profile(profile),
                }
            }
        )
    except RuntimeError as exc:
        logger.warning("resume_upload runtime error: %s", exc)
        return error_json(str(exc), status=400)
    except Exception:
        logger.exception("resume_upload")
        return error_json("简历解析失败，请稍后重试。", status=500)


async def resume_save(request):
    try:
        user = get_user_from_request(request)
        if not user:
            return web.json_response({"code": -1, "msg": "请先登录"}, status=401)

        params = await request.json()
        profile = _sanitize_resume_profile(params.get("profile") or {})
        file_name = str(params.get("file_name", "") or "").strip()[:200]
        job_title = str(params.get("job_title", "") or "").strip()[:120]

        save_user_resume(user["id"], file_name, profile, job_title)
        return ok_json({"msg": "简历已保存"})
    except Exception as e:
        logger.exception("resume_save")
        return error_json(str(e), status=500)


async def resume_mine(request):
    try:
        user = get_user_from_request(request)
        if not user:
            return web.json_response({"code": -1, "msg": "请先登录"}, status=401)

        resume = get_user_resume(user["id"])
        return ok_json({"data": resume})
    except Exception as e:
        logger.exception("resume_mine")
        return error_json(str(e), status=500)


async def asr_transcribe(request):
    try:
        form = await request.post()
        fileobj = form.get("file")
        if fileobj is None:
            return error_json("缺少语音文件。", status=400)

        filename = getattr(fileobj, "filename", "") or "voice.webm"
        filebytes = fileobj.file.read()
        if not filebytes:
            return error_json("上传的语音文件为空。", status=400)

        wav_bytes = await asyncio.to_thread(_convert_audio_to_wav_bytes, filebytes, filename)
        text = await _tencent_sentence_recognition(wav_bytes)
        return ok_json({"data": {"text": text}})
    except RuntimeError as exc:
        logger.warning("asr_transcribe runtime error: %s", exc)
        return error_json(str(exc), status=400)
    except Exception as exc:
        logger.exception("asr_transcribe")
        return error_json(str(exc), status=500)


async def set_audiotype(request):
    try:
        params = await request.json()

        sessionid = int(params.get('sessionid',0))
        get_session_real(sessionid).set_custom_state(params['audiotype'],params['reinit'])
        return ok_json({"msg":"ok"})
    except Exception as e:
        logger.exception('exception:')
        return error_json(str(e), status=500)

async def record(request):
    try:
        params = await request.json()

        sessionid = int(params.get('sessionid',0))
        nerfreal = get_session_real(sessionid)
        if params['type']=='start_record':
            # nerfreals[sessionid].put_msg_txt(params['text'])
            nerfreal.start_recording()
        elif params['type']=='end_record':
            nerfreal.stop_recording()
        return ok_json({"msg":"ok"})
    except Exception as e:
        logger.exception('exception:')
        return error_json(str(e), status=500)

async def is_speaking(request):
    params = await request.json()

    sessionid = int(params.get('sessionid',0))
    return ok_json({"data": get_session_real(sessionid).is_speaking()})


async def interviewer_options(request):
    options = [_serialize_interviewer_option(option) for option in get_interviewer_options()]
    return ok_json({"data": {"default_id": DEFAULT_INTERVIEWER_ID, "options": options}})


async def interviewer_preview(request):
    interviewer_id = request.match_info.get("interviewer_id", "")
    option = get_interviewer_option(interviewer_id)
    preview_path = _find_avatar_preview_path(option["avatar_id"])
    if preview_path is None or not preview_path.exists():
        return error_json("未找到面试官预览图。", status=404)
    return web.FileResponse(path=preview_path)


async def interview_start(request):
    try:
        user = get_user_from_request(request)
        if not user:
            return web.json_response({"code": -1, "msg": "请先登录"}, status=401)
        if user['free_quota'] + user['paid_quota'] <= 0:
            return web.json_response({"code": -1, "msg": "面试次数已用完，请充值后继续"}, status=403)
        db_deduct_quota(user['id'])

        params = await request.json()
        sessionid = int(params.get('sessionid', 0))
        job_title = str(params.get('job_title', '')).strip()
        job_requirements = _sanitize_text_block(params.get('job_requirements', ''))
        interview_mode = str(params.get('interview_mode', 'text')).strip()
        interviewer_id = str(params.get('interviewer_id', '') or '').strip()
        resume_file_name = str(params.get('resume_file_name', '') or '').strip()
        resume_profile = _sanitize_resume_profile(params.get('resume_profile'))

        nerfreal = get_session_real(sessionid)
        interviewer = get_interviewer_option(interviewer_id or getattr(nerfreal.opt, "interviewer_id", DEFAULT_INTERVIEWER_ID))
        localized_job_title = translate_job_title(job_title, interviewer["language"])
        nerfreal.reset_interview_state(keep_context=False)
        nerfreal.set_interview_context(
            job_title=localized_job_title,
            job_requirements=job_requirements,
            interview_mode=interview_mode,
            resume_file_name=resume_file_name,
            resume_profile=resume_profile,
            interviewer_id=interviewer["id"],
            interviewer_name=interviewer["display_name"],
            interviewer_language=interviewer["language"],
            interviewer_language_label=interviewer["language_label"],
        )

        outline = await asyncio.get_event_loop().run_in_executor(None, generate_interview_outline, nerfreal)
        outline_progress = build_initial_outline_progress(outline)
        nerfreal.set_interview_context(
            interview_outline=outline,
            outline_progress=outline_progress,
            interview_started_at=time.time(),
            interview_finished=False,
            interview_finish_reason="",
            interview_report_ready_at=0.0,
            wrap_up_turns=0,
        )
        outline_log_context = nerfreal.get_interview_context()
        outline_log_context["sessionid"] = sessionid
        outline_log = format_interview_outline_for_log(outline, outline_log_context)
        logger.info("\n%s", outline_log)

        opening = build_interview_opening(nerfreal)
        nerfreal.add_interview_message('assistant', opening, {'source': 'opening'})
        nerfreal.put_msg_txt(opening, {'source': 'opening'})

        return ok_json(
            {
                "msg": "ok",
                "opening": opening,
                "context": nerfreal.get_interview_context(),
            }
        )
    except Exception as e:
        logger.exception('interview_start')
        return error_json(str(e), status=500)


async def interview_state(request):
    try:
        params = await request.json()
        sessionid = int(params.get('sessionid', 0))
        last_id = int(params.get('last_id', 0))
        if sessionid <= 0:
            return ok_json({"data": build_invalid_session_state("empty_session")})

        nerfreal = nerfreals.get(sessionid)
        if nerfreal is None:
            return ok_json({"data": build_invalid_session_state("session_not_found")})

        nerfreal = get_session_real(sessionid)
        return ok_json({"data": nerfreal.get_interview_state(last_id)})
    except Exception as e:
        logger.exception('interview_state')
        return error_json(str(e), status=500)


async def interview_report(request):
    try:
        params = await request.json()
        sessionid = int(params.get('sessionid', 0))
        nerfreal = get_session_real(sessionid)
        nerfreal.flush_talk()
        report = await asyncio.get_event_loop().run_in_executor(None, generate_interview_report, nerfreal)

        user = get_user_from_request(request)
        if user and report:
            ctx = nerfreal.get_interview_context()
            try:
                save_interview_report(
                    user_id=user["id"],
                    job_title=ctx.get("job_title", ""),
                    interview_mode=ctx.get("interview_mode", "text"),
                    score=int(report.get("score", 0)),
                    recommendation=str(report.get("recommendation", "")),
                    report_dict=report,
                )
            except Exception:
                logger.exception("save_interview_report failed (non-fatal)")

        return ok_json({"data": report})
    except Exception as e:
        logger.exception('interview_report')
        return error_json(str(e), status=500)


async def interview_reports_list(request):
    try:
        user = get_user_from_request(request)
        if not user:
            return web.json_response({"code": -1, "msg": "请先登录"}, status=401)

        reports = get_user_interview_reports(user["id"])
        return ok_json({"data": {"reports": reports}})
    except Exception as e:
        logger.exception("interview_reports_list")
        return error_json(str(e), status=500)


async def interview_reset(request):
    try:
        params = await request.json()
        sessionid = int(params.get('sessionid', 0))
        keep_context = bool(params.get('keep_context', True))
        nerfreal = get_session_real(sessionid)
        nerfreal.reset_interview_state(keep_context=keep_context)
        if params.get('job_title') or params.get('job_requirements') or params.get('interview_mode') or params.get('interviewer_id'):
            interviewer = get_interviewer_option(
                str(params.get('interviewer_id', '') or '') or nerfreal.get_interview_context().get('interviewer_id', DEFAULT_INTERVIEWER_ID)
            )
            next_job_title = str(params.get('job_title', '')).strip() or nerfreal.get_interview_context()['job_title']
            next_job_requirements = _sanitize_text_block(params.get('job_requirements', '')) or nerfreal.get_interview_context().get('job_requirements', '')
            nerfreal.set_interview_context(
                job_title=translate_job_title(next_job_title, interviewer["language"]),
                job_requirements=next_job_requirements,
                interview_mode=str(params.get('interview_mode', nerfreal.get_interview_context()['interview_mode'])).strip(),
                interviewer_id=interviewer["id"],
                interviewer_name=interviewer["display_name"],
                interviewer_language=interviewer["language"],
                interviewer_language_label=interviewer["language_label"],
            )
        return ok_json({"msg": "ok", "context": nerfreal.get_interview_context()})
    except Exception as e:
        logger.exception('interview_reset')
        return error_json(str(e), status=500)


async def interview_terminate(request):
    try:
        params = await request.json()
        delay_seconds = max(0.5, float(params.get('delay_seconds', 1.0)))

        def _shutdown_server():
            #os._exit(0)
            pass
        
        Timer(delay_seconds, _shutdown_server).start()
        return ok_json({"msg": "scheduled", "delay_seconds": delay_seconds})
    except Exception as e:
        logger.exception('interview_terminate')
        return error_json(str(e), status=500)


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

async def post(url,data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url,data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        logger.info(f'Error: {e}')

async def run(push_url,sessionid):
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
    nerfreals[sessionid] = nerfreal

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url,pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer,type='answer'))
##########################################
# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'                                                    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    
    # audio FPS
    parser.add_argument('--fps', type=int, default=50, help="audio fps,must be 50")
    # sliding window left-middle-right length (unit: 20ms)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)

    parser.add_argument('--W', type=int, default=450, help="GUI width")
    parser.add_argument('--H', type=int, default=450, help="GUI height")

    #musetalk opt
    parser.add_argument('--avatar_id', type=str, default='avator_1', help="define which avatar in data/avatars")
    #parser.add_argument('--bbox_shift', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16, help="infer batch")

    parser.add_argument('--customvideo_config', type=str, default='', help="custom action json")

    parser.add_argument('--tts', type=str, default='edgetts', help="tts service type") #xtts gpt-sovits cosyvoice fishtts tencent doubao indextts2 azuretts
    parser.add_argument('--REF_FILE', type=str, default="zh-CN-YunxiaNeural",help="参考文件名或语音模型ID，默认值为 edgetts的语音模型ID zh-CN-YunxiaNeural, 若--tts指定为azuretts, 可以使用Azure语音模型ID, 如zh-CN-XiaoxiaoMultilingualNeural")
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880') # http://localhost:9000
    # parser.add_argument('--CHARACTER', type=str, default='test')
    # parser.add_argument('--EMOTION', type=str, default='default')

    parser.add_argument('--model', type=str, default='musetalk') #musetalk wav2lip ultralight

    parser.add_argument('--transport', type=str, default='rtcpush') #webrtc rtcpush virtualcam
    parser.add_argument('--push_url', type=str, default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream') #rtmp://localhost/live/livestream

    parser.add_argument('--max_session', type=int, default=1)  #multi session count
    parser.add_argument('--listenport', type=int, default=8010, help="web listen port")

    opt = parser.parse_args()
    opt = apply_env_overrides(opt)
    #app.config.from_object(opt)
    #print(app.config)
    opt.customopt = []
    if opt.customvideo_config!='':
        with open(opt.customvideo_config,'r') as file:
            opt.customopt = json.load(file)

    # if opt.model == 'ernerf':       
    #     from nerfreal import NeRFReal,load_model,load_avatar
    #     model = load_model(opt)
    #     avatar = load_avatar(opt) 
    if opt.model == 'musetalk':
        from musereal import MuseReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model()
        avatar = load_avatar(opt.avatar_id) 
        warm_up(opt.batch_size,model)      
    elif opt.model == 'wav2lip':
        from lipreal import LipReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model("./models/wav2lip.pth")
        avatar = None
        warm_up(opt.batch_size,model,256)
    elif opt.model == 'ultralight':
        from lightreal import LightReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model(opt)
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,avatar,160)

    # if opt.transport=='rtmp':
    #     thread_quit = Event()
    #     nerfreals[0] = build_nerfreal(0)
    #     rendthrd = Thread(target=nerfreals[0].render,args=(thread_quit,))
    #     rendthrd.start()
    if opt.transport=='virtualcam':
        thread_quit = Event()
        nerfreals[0] = build_nerfreal(0)
        rendthrd = Thread(target=nerfreals[0].render,args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    appasync = web.Application(client_max_size=1024**2*100)
    appasync.on_shutdown.append(on_shutdown)
    setup_auth_routes(appasync)
    setup_payment_routes(appasync)

    async def sms_mode(request):
        from auth import SMS_DEV_MODE
        return web.json_response({"dev_mode": SMS_DEV_MODE})
    appasync.router.add_get("/auth/sms-mode", sms_mode)
    appasync.router.add_get("/interviewer/options", interviewer_options)
    appasync.router.add_get("/interviewer/preview/{interviewer_id}", interviewer_preview)
    appasync.router.add_post("/interviewer/prepare", interviewer_prepare)
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/human", human)
    appasync.router.add_post("/humanaudio", humanaudio)
    appasync.router.add_post("/resume/upload", resume_upload)
    appasync.router.add_post("/resume/save", resume_save)
    appasync.router.add_get("/resume/mine", resume_mine)
    appasync.router.add_post("/asr/transcribe", asr_transcribe)
    appasync.router.add_post("/set_audiotype", set_audiotype)
    appasync.router.add_post("/record", record)
    appasync.router.add_post("/interrupt_talk", interrupt_talk)
    appasync.router.add_post("/is_speaking", is_speaking)
    appasync.router.add_post("/interview/start", interview_start)
    appasync.router.add_post("/interview/state", interview_state)
    appasync.router.add_post("/interview/report", interview_report)
    appasync.router.add_get("/interview/reports", interview_reports_list)
    appasync.router.add_post("/interview/reset", interview_reset)
    appasync.router.add_post("/interview/terminate", interview_terminate)
    appasync.router.add_static('/',path='web')

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    # Configure CORS on all routes.
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename='interview.html'
    if opt.transport=='rtmp':
        pagename='echoapi.html'
    elif opt.transport=='rtcpush':
        pagename='rtcpushapi.html'
    logger.info('start http server; http://<serverip>:'+str(opt.listenport)+'/'+pagename)
    logger.info('如果使用webrtc，推荐访问面试前端: http://<serverip>:'+str(opt.listenport)+'/interview.html')
    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '127.0.0.1', opt.listenport)
        loop.run_until_complete(site.start())
        if opt.transport=='rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k!=0:
                    push_url = opt.push_url+str(k)
                loop.run_until_complete(run(push_url,k))
        loop.run_forever()    
    #Thread(target=run_server, args=(web.AppRunner(appasync),)).start()
    run_server(web.AppRunner(appasync))

    #app.on_shutdown.append(on_shutdown)
    #app.router.add_post("/offer", offer)

    # print('start websocket server')
    # server = pywsgi.WSGIServer(('0.0.0.0', 8000), app, handler_class=WebSocketHandler)
    # server.serve_forever()
    
    
