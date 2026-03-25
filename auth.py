import os
import time
import re
from aiohttp import web
import jwt

from user_db import get_or_create_user, get_user_by_id, get_user_quota
from logger import logger

JWT_SECRET = os.getenv("JWT_SECRET", "livetalking-dev-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_SECONDS = 7 * 24 * 3600

_verification_codes: dict[str, dict] = {}

PHONE_RE = re.compile(r"^1[3-9]\d{9}$")


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
    user = get_user_by_id(payload["user_id"])
    return user


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
            return web.json_response({"code": -1, "msg": "问题额度已用完，请充值后继续"}, status=403)
        request["user"] = user
        return await handler(request)
    return wrapper


async def send_code(request):
    try:
        params = await request.json()
        phone = str(params.get("phone", "")).strip()
        if not phone or not PHONE_RE.match(phone):
            return web.json_response({"code": -1, "msg": "请输入正确的手机号"}, status=400)

        code = "1234"
        _verification_codes[phone] = {
            "code": code,
            "expires_at": time.time() + 300,
        }
        logger.info("验证码已发送到 %s: %s (开发模式固定验证码)", phone, code)
        return web.json_response({"code": 0, "msg": "验证码已发送"})
    except Exception as e:
        logger.exception("send_code error")
        return web.json_response({"code": -1, "msg": str(e)}, status=500)


async def verify_code(request):
    try:
        params = await request.json()
        phone = str(params.get("phone", "")).strip()
        code = str(params.get("code", "")).strip()

        if not phone or not PHONE_RE.match(phone):
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
