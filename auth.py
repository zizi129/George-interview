import os
import random
import string
import time
import re
from aiohttp import web
import jwt

from user_db import get_or_create_user, get_user_by_id, get_user_quota
from logger import logger

JWT_SECRET = os.getenv("JWT_SECRET", "livetalking-dev-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_SECONDS = 7 * 24 * 3600

DEV_MODE = os.getenv("SMS_DEV_MODE", "false").lower() in ("1", "true", "yes")
SMS_DEV_MODE = DEV_MODE

# ── 腾讯云短信配置 ────────────────────────────────────────────────────────────
TENCENT_SECRET_ID = os.getenv("TENCENT_SECRET_ID", "")
TENCENT_SECRET_KEY = os.getenv("TENCENT_SECRET_KEY", "")
TENCENT_SMS_APPID = os.getenv("TENCENT_SMS_APPID", "")
TENCENT_SMS_SIGN_NAME = os.getenv("TENCENT_SMS_SIGN_NAME", "")
TENCENT_SMS_TEMPLATE_ID = os.getenv("TENCENT_SMS_TEMPLATE_ID", "")

_verification_codes: dict[str, dict] = {}

PHONE_RE = re.compile(r"^1[3-9]\d{9}$")


def _is_phone(phone: str) -> bool:
    return bool(PHONE_RE.match(phone))


def _generate_code(length: int = 6) -> str:
    return "".join(random.choices(string.digits, k=length))


# ── 腾讯云短信发送 ─────────────────────────────────────────────────────────────

def _send_sms_tencent(phone: str, code: str) -> None:
    try:
        from tencentcloud.common import credential
        from tencentcloud.sms.v20210111 import sms_client, models
    except ImportError as e:
        raise RuntimeError("腾讯云 SMS SDK 未安装：pip install tencentcloud-sdk-python-sms") from e

    if not all([TENCENT_SECRET_ID, TENCENT_SECRET_KEY, TENCENT_SMS_APPID,
                TENCENT_SMS_SIGN_NAME, TENCENT_SMS_TEMPLATE_ID]):
        raise RuntimeError("腾讯云短信配置不完整，请检查 .env 中的 TENCENT_SMS_* 变量")

    cred = credential.Credential(TENCENT_SECRET_ID, TENCENT_SECRET_KEY)
    client = sms_client.SmsClient(cred, "ap-guangzhou")
    req = models.SendSmsRequest()
    req.SmsSdkAppId = TENCENT_SMS_APPID
    req.SignName = TENCENT_SMS_SIGN_NAME
    req.TemplateId = TENCENT_SMS_TEMPLATE_ID
    req.TemplateParamSet = [code, "5"]
    req.PhoneNumberSet = [f"+86{phone}"]
    resp = client.SendSms(req)
    status = resp.SendStatusSet[0] if resp.SendStatusSet else None
    if not status or status.Code != "Ok":
        raise RuntimeError(f"腾讯云短信发送失败: {getattr(status, 'Code', '?')} {getattr(status, 'Message', '')}")
    logger.info("腾讯云短信已发送到 %s", phone)


def _send_code(phone: str, code: str) -> None:
    if DEV_MODE:
        logger.warning("【开发模式】验证码 %s -> %s（未真实发送）", phone, code)
        return

    _send_sms_tencent(phone, code)


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_token(user_id: int, phone: str) -> str:
    payload = {
        "user_id": user_id,
        "phone": phone,
        "exp": time.time() + JWT_EXPIRE_SECONDS,
        "iat": time.time(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except (jwt.InvalidTokenError, jwt.ExpiredSignatureError):
        return None


def get_user_from_request(request) -> dict | None:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header[7:].strip()
    if not token:
        return None
    payload = decode_token(token)
    if not payload:
        return None
    return get_user_by_id(payload["user_id"])


def require_auth(handler):
    async def wrapper(request):
        user = get_user_from_request(request)
        if not user:
            return web.json_response({"code": -1, "msg": "请先登录"}, status=401)
        request["user"] = user
        return await handler(request)
    return wrapper


def require_quota(handler):
    async def wrapper(request):
        user = get_user_from_request(request)
        if not user:
            return web.json_response({"code": -1, "msg": "请先登录"}, status=401)
        if user["free_quota"] + user["paid_quota"] <= 0:
            return web.json_response({"code": -1, "msg": "面试次数已用完，请充值后继续"}, status=403)
        request["user"] = user
        return await handler(request)
    return wrapper


# ── API 处理器 ────────────────────────────────────────────────────────────────

async def send_code(request):
    try:
        params = await request.json()
        phone = str(params.get("account") or params.get("phone", "")).strip()

        if not phone:
            return web.json_response({"code": -1, "msg": "请输入手机号"}, status=400)
        if not _is_phone(phone):
            return web.json_response({"code": -1, "msg": "请输入正确的手机号"}, status=400)

        existing = _verification_codes.get(phone)
        if existing and existing.get("sent_at", 0) + 60 > time.time():
            remaining = int(existing["sent_at"] + 60 - time.time())
            return web.json_response({"code": -1, "msg": f"请 {remaining} 秒后再试"}, status=429)

        code = _generate_code(6)

        try:
            _send_code(phone, code)
        except Exception as err:
            logger.error("验证码发送失败 phone=%s: %s", phone, err)
            return web.json_response({"code": -1, "msg": f"发送失败：{err}"}, status=500)

        _verification_codes[phone] = {
            "code": code,
            "expires_at": time.time() + 300,
            "sent_at": time.time(),
        }

        return web.json_response({"code": 0, "msg": "验证码已发送"})
    except Exception as e:
        logger.exception("send_code error")
        return web.json_response({"code": -1, "msg": str(e)}, status=500)


async def verify_code(request):
    try:
        params = await request.json()
        phone = str(params.get("account") or params.get("phone", "")).strip()
        code = str(params.get("code", "")).strip()

        if not phone:
            return web.json_response({"code": -1, "msg": "请输入手机号"}, status=400)
        if not _is_phone(phone):
            return web.json_response({"code": -1, "msg": "请输入正确的手机号"}, status=400)
        if not code:
            return web.json_response({"code": -1, "msg": "请输入验证码"}, status=400)

        stored = _verification_codes.get(phone)
        if not stored or stored["code"] != code or stored["expires_at"] < time.time():
            return web.json_response({"code": -1, "msg": "验证码错误或已过期"}, status=400)

        _verification_codes.pop(phone, None)
        user = get_or_create_user(phone)
        token = create_token(user["id"], user["phone"])

        return web.json_response({
            "code": 0,
            "data": {
                "token": token,
                "user": {
                    "id": user["id"],
                    "phone": user["phone"],
                    "nickname": user["nickname"],
                    "free_quota": user["free_quota"],
                    "paid_quota": user["paid_quota"],
                },
            },
        })
    except Exception as e:
        logger.exception("verify_code error")
        return web.json_response({"code": -1, "msg": str(e)}, status=500)


async def get_me(request):
    try:
        user = get_user_from_request(request)
        if not user:
            return web.json_response({"code": -1, "msg": "请先登录"}, status=401)

        quota = get_user_quota(user["id"])
        return web.json_response({
            "code": 0,
            "data": {
                "id": user["id"],
                "phone": user["phone"],
                "nickname": user["nickname"],
                "free_quota": quota["free_quota"],
                "paid_quota": quota["paid_quota"],
                "total_quota": quota["total"],
            },
        })
    except Exception as e:
        logger.exception("get_me error")
        return web.json_response({"code": -1, "msg": str(e)}, status=500)


def setup_auth_routes(app):
    app.router.add_post("/auth/send-code", send_code)
    app.router.add_post("/auth/verify-code", verify_code)
    app.router.add_get("/auth/me", get_me)
