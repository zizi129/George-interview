import os
import random
import string
import time
import re
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from aiohttp import web
import jwt

from user_db import get_or_create_user, get_user_by_id, get_user_quota
from logger import logger

JWT_SECRET = os.getenv("JWT_SECRET", "livetalking-dev-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_SECONDS = 7 * 24 * 3600

# 开发模式：不真实发送，验证码只打印到日志
DEV_MODE = os.getenv("SMS_DEV_MODE", "false").lower() in ("1", "true", "yes")
# 兼容旧调用点，避免 app.py 等模块导入失败
SMS_DEV_MODE = DEV_MODE

# 验证渠道：email（默认）或 sms
AUTH_CHANNEL = os.getenv("AUTH_CHANNEL", "email").lower()

# ── 邮箱 SMTP 配置 ────────────────────────────────────────────────────────────
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.qq.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")          # 发件邮箱，如 xxx@qq.com
SMTP_PASS = os.getenv("SMTP_PASS", "")          # QQ邮箱授权码 / 其他邮箱密码
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "乔治面试")
EMAIL_SUBJECT = os.getenv("EMAIL_SUBJECT", "【乔治面试】您的登录验证码")

# ── 腾讯云短信配置（AUTH_CHANNEL=sms 时使用）────────────────────────────────
TENCENT_SECRET_ID = os.getenv("TENCENT_SECRET_ID", "")
TENCENT_SECRET_KEY = os.getenv("TENCENT_SECRET_KEY", "")
TENCENT_SMS_APPID = os.getenv("TENCENT_SMS_APPID", "")
TENCENT_SMS_SIGN_NAME = os.getenv("TENCENT_SMS_SIGN_NAME", "")
TENCENT_SMS_TEMPLATE_ID = os.getenv("TENCENT_SMS_TEMPLATE_ID", "")

# ── 阿里云短信配置（AUTH_CHANNEL=sms, SMS_BACKEND=aliyun 时使用）─────────────
SMS_BACKEND = os.getenv("SMS_BACKEND", "tencent").lower()
ALIYUN_ACCESS_KEY_ID = os.getenv("ALIYUN_ACCESS_KEY_ID", "")
ALIYUN_ACCESS_KEY_SECRET = os.getenv("ALIYUN_ACCESS_KEY_SECRET", "")
ALIYUN_SMS_SIGN_NAME = os.getenv("ALIYUN_SMS_SIGN_NAME", "")
ALIYUN_SMS_TEMPLATE_CODE = os.getenv("ALIYUN_SMS_TEMPLATE_CODE", "")

_verification_codes: dict[str, dict] = {}

PHONE_RE = re.compile(r"^1[3-9]\d{9}$")
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def _is_email(account: str) -> bool:
    return "@" in account and bool(EMAIL_RE.match(account))


def _is_phone(account: str) -> bool:
    return bool(PHONE_RE.match(account))


def _generate_code(length: int = 6) -> str:
    return "".join(random.choices(string.digits, k=length))


# ── 邮箱发送 ─────────────────────────────────────────────────────────────────

def _send_email(to_addr: str, code: str) -> None:
    """通过 SMTP 发送验证码邮件（同步，在线程池中调用）"""
    if not SMTP_USER or not SMTP_PASS:
        raise RuntimeError(
            "邮箱 SMTP 未配置，请在 .env 中设置 SMTP_USER 和 SMTP_PASS"
        )

    body = (
        f"您好，\n\n"
        f"您的登录验证码为：{code}\n\n"
        f"验证码 5 分钟内有效，请勿泄露给他人。\n\n"
        f"— 乔治面试团队"
    )
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = Header(EMAIL_SUBJECT, "utf-8")
    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_USER}>"
    msg["To"] = to_addr

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, [to_addr], msg.as_string())

    logger.info("验证码邮件已发送到 %s", to_addr)


# ── 短信发送 ─────────────────────────────────────────────────────────────────

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


def _send_sms_aliyun(phone: str, code: str) -> None:
    try:
        from alibabacloud_dysmsapi20170525.client import Client
        from alibabacloud_dysmsapi20170525 import models as sms_models
        from alibabacloud_tea_openapi import models as open_api_models
    except ImportError as e:
        raise RuntimeError("阿里云 SMS SDK 未安装：pip install alibabacloud-dysmsapi20170525") from e

    if not all([ALIYUN_ACCESS_KEY_ID, ALIYUN_ACCESS_KEY_SECRET,
                ALIYUN_SMS_SIGN_NAME, ALIYUN_SMS_TEMPLATE_CODE]):
        raise RuntimeError("阿里云短信配置不完整，请检查 .env 中的 ALIYUN_* 变量")

    import json
    config = open_api_models.Config(
        access_key_id=ALIYUN_ACCESS_KEY_ID,
        access_key_secret=ALIYUN_ACCESS_KEY_SECRET,
    )
    config.endpoint = "dysmsapi.aliyuncs.com"
    client = Client(config)
    req = sms_models.SendSmsRequest(
        phone_numbers=phone,
        sign_name=ALIYUN_SMS_SIGN_NAME,
        template_code=ALIYUN_SMS_TEMPLATE_CODE,
        template_param=json.dumps({"code": code}),
    )
    resp = client.send_sms(req)
    if resp.body.code != "OK":
        raise RuntimeError(f"阿里云短信发送失败: {resp.body.code} {resp.body.message}")
    logger.info("阿里云短信已发送到 %s", phone)


# ── 统一发送入口 ──────────────────────────────────────────────────────────────

def _send_code(account: str, code: str) -> None:
    """根据账号类型和配置选择发送渠道"""
    if DEV_MODE:
        logger.warning("【开发模式】验证码 %s -> %s（未真实发送）", account, code)
        return

    if _is_email(account):
        _send_email(account, code)
    elif AUTH_CHANNEL == "sms":
        if SMS_BACKEND == "aliyun":
            _send_sms_aliyun(account, code)
        else:
            _send_sms_tencent(account, code)
    else:
        raise RuntimeError(
            "该账号为手机号，但当前未配置短信服务（AUTH_CHANNEL=email）。"
            "请使用邮箱登录，或配置短信服务后设置 AUTH_CHANNEL=sms"
        )


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
        # 兼容旧字段 phone，也接受新字段 account
        account = str(params.get("account") or params.get("phone", "")).strip()

        if not account:
            return web.json_response({"code": -1, "msg": "请输入手机号或邮箱"}, status=400)
        if not _is_email(account) and not _is_phone(account):
            return web.json_response({"code": -1, "msg": "请输入正确的手机号或邮箱地址"}, status=400)

        # 频率限制：60 秒内只能发一次
        existing = _verification_codes.get(account)
        if existing and existing.get("sent_at", 0) + 60 > time.time():
            remaining = int(existing["sent_at"] + 60 - time.time())
            return web.json_response({"code": -1, "msg": f"请 {remaining} 秒后再试"}, status=429)

        code = _generate_code(6)

        try:
            _send_code(account, code)
        except Exception as err:
            logger.error("验证码发送失败 account=%s: %s", account, err)
            return web.json_response({"code": -1, "msg": f"发送失败：{err}"}, status=500)

        _verification_codes[account] = {
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
        account = str(params.get("account") or params.get("phone", "")).strip()
        code = str(params.get("code", "")).strip()

        if not account:
            return web.json_response({"code": -1, "msg": "请输入手机号或邮箱"}, status=400)
        if not _is_email(account) and not _is_phone(account):
            return web.json_response({"code": -1, "msg": "请输入正确的手机号或邮箱地址"}, status=400)
        if not code:
            return web.json_response({"code": -1, "msg": "请输入验证码"}, status=400)

        stored = _verification_codes.get(account)
        if not stored or stored["code"] != code or stored["expires_at"] < time.time():
            return web.json_response({"code": -1, "msg": "验证码错误或已过期"}, status=400)

        _verification_codes.pop(account, None)
        user = get_or_create_user(account)
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
