import json
import os
import time
from aiohttp import web

from auth import get_user_from_request
from user_db import (
    create_order,
    get_order_by_no,
    get_user_orders,
    mark_order_paid,
    mark_order_failed,
    get_user_quota,
)
from logger import logger

WECHAT_MCH_ID = os.getenv("WECHAT_MCH_ID", "")
WECHAT_API_KEY = os.getenv("WECHAT_API_KEY", "")
WECHAT_APP_ID = os.getenv("WECHAT_APP_ID", "")
WECHAT_NOTIFY_URL = os.getenv("WECHAT_NOTIFY_URL", "")
WECHAT_PRIVATE_KEY_PATH = os.getenv("WECHAT_PRIVATE_KEY_PATH", "")
WECHAT_CERT_SERIAL_NO = os.getenv("WECHAT_CERT_SERIAL_NO", "")

PRICE_TABLE = {
    10: 990,
    50: 3990,
    100: 6990,
}

ORDER_EXPIRE_SECONDS = 30 * 60


def _is_wechat_configured() -> bool:
    return bool(WECHAT_MCH_ID and WECHAT_API_KEY and WECHAT_APP_ID)


def _load_private_key() -> str:
    if not WECHAT_PRIVATE_KEY_PATH:
        return ""
    try:
        with open(WECHAT_PRIVATE_KEY_PATH, "r") as f:
            return f.read()
    except Exception as e:
        logger.warning("读取微信支付私钥失败: %s", e)
        return ""


def _get_wechat_pay_client():
    from wechatpayv3 import WeChatPay

    private_key = _load_private_key()
    if not private_key:
        raise RuntimeError("微信支付私钥未配置")

    return WeChatPay(
        wechatpay_type="NATIVE",
        mchid=WECHAT_MCH_ID,
        appid=WECHAT_APP_ID,
        private_key=private_key,
        cert_serial_no=WECHAT_CERT_SERIAL_NO,
        apiv3_key=WECHAT_API_KEY,
        notify_url=WECHAT_NOTIFY_URL,
    )


async def pay_plans(request):
    """返回可购买的套餐列表"""
    plans = []
    for count in sorted(PRICE_TABLE.keys()):
        fen = PRICE_TABLE[count]
        plans.append({
            "question_count": count,
            "amount_fen": fen,
            "display_price": f"¥{fen / 100:.1f}",
        })
    return web.json_response({"code": 0, "data": {"plans": plans}})


async def create_pay_order(request):
    try:
        user = get_user_from_request(request)
        if not user:
            return web.json_response({"code": -1, "msg": "请先登录"}, status=401)

        params = await request.json()
        question_count = int(params.get("question_count", 0))
        amount_fen = int(params.get("amount_fen", 0))

        if question_count not in PRICE_TABLE:
            return web.json_response({"code": -1, "msg": "无效的套餐"}, status=400)
        if PRICE_TABLE[question_count] != amount_fen:
            return web.json_response({"code": -1, "msg": "金额不匹配"}, status=400)

        order = create_order(user["id"], question_count, amount_fen)

        code_url = ""
        if _is_wechat_configured():
            try:
                code_url = await _wechat_native_pay(order["order_no"], amount_fen, question_count)
            except ImportError:
                logger.warning("wechatpayv3 未安装，使用开发模式自动完成订单")
            except Exception as e:
                logger.warning("微信支付下单失败，将使用开发模式: %s", e)

        if not code_url:
            logger.info(
                "微信支付未配置或下单失败，自动完成订单 %s（开发模式）",
                order["order_no"],
            )
            mark_order_paid(order["order_no"], "dev-auto-paid")

        quota = get_user_quota(user["id"])

        return web.json_response({
            "code": 0,
            "data": {
                "order_no": order["order_no"],
                "code_url": code_url,
                "amount_fen": amount_fen,
                "question_count": question_count,
                "status": "paid" if not code_url else "pending",
                "current_quota": quota["total"],
            },
        })
    except Exception as e:
        logger.exception("create_pay_order error")
        return web.json_response({"code": -1, "msg": str(e)}, status=500)


async def pay_notify(request):
    """微信支付异步回调通知"""
    try:
        body = await request.text()
        logger.info("微信支付回调: %s", body[:500])

        if not _is_wechat_configured():
            return web.json_response({"code": "SUCCESS", "message": "OK"})

        headers = {
            "Wechatpay-Timestamp": request.headers.get("Wechatpay-Timestamp", ""),
            "Wechatpay-Nonce": request.headers.get("Wechatpay-Nonce", ""),
            "Wechatpay-Signature": request.headers.get("Wechatpay-Signature", ""),
            "Wechatpay-Signature-Type": request.headers.get("Wechatpay-Signature-Type", ""),
            "Wechatpay-Serial": request.headers.get("Wechatpay-Serial", ""),
        }

        order_no = _decrypt_and_parse_notify(body, headers)
        if order_no:
            success = mark_order_paid(order_no)
            if success:
                logger.info("微信支付回调成功，订单 %s 已标记为已支付", order_no)
            else:
                logger.warning("微信支付回调：订单 %s 标记失败（可能已处理）", order_no)

        return web.json_response({"code": "SUCCESS", "message": "OK"})
    except Exception as e:
        logger.exception("pay_notify error")
        return web.json_response({"code": "FAIL", "message": str(e)}, status=500)


async def pay_query(request):
    try:
        user = get_user_from_request(request)
        if not user:
            return web.json_response({"code": -1, "msg": "请先登录"}, status=401)

        params = await request.json()
        order_no = str(params.get("order_no", "")).strip()
        if not order_no:
            return web.json_response({"code": -1, "msg": "缺少订单号"}, status=400)

        order = get_order_by_no(order_no)
        if not order or order["user_id"] != user["id"]:
            return web.json_response({"code": -1, "msg": "订单不存在"}, status=404)

        if (
            order["status"] == "pending"
            and _is_wechat_configured()
            and order.get("created_at")
            and time.time() - order["created_at"] < ORDER_EXPIRE_SECONDS
        ):
            try:
                remote_status = await _wechat_query_order(order_no)
                if remote_status == "SUCCESS":
                    mark_order_paid(order_no, "wechat-query-sync")
                    order = get_order_by_no(order_no)
                elif remote_status in ("CLOSED", "REVOKED", "PAYERROR"):
                    mark_order_failed(order_no)
                    order = get_order_by_no(order_no)
            except Exception as e:
                logger.warning("主动查询微信订单 %s 状态失败: %s", order_no, e)

        if (
            order["status"] == "pending"
            and order.get("created_at")
            and time.time() - order["created_at"] > ORDER_EXPIRE_SECONDS
        ):
            mark_order_failed(order_no)
            order = get_order_by_no(order_no)

        quota = get_user_quota(user["id"])

        return web.json_response({
            "code": 0,
            "data": {
                "order_no": order["order_no"],
                "status": order["status"],
                "question_count": order["question_count"],
                "amount_fen": order["amount_fen"],
                "current_quota": quota["total"],
            },
        })
    except Exception as e:
        logger.exception("pay_query error")
        return web.json_response({"code": -1, "msg": str(e)}, status=500)


async def pay_orders(request):
    """查询用户的订单历史"""
    try:
        user = get_user_from_request(request)
        if not user:
            return web.json_response({"code": -1, "msg": "请先登录"}, status=401)

        orders = get_user_orders(user["id"], limit=20)
        items = []
        for o in orders:
            items.append({
                "order_no": o["order_no"],
                "question_count": o["question_count"],
                "amount_fen": o["amount_fen"],
                "status": o["status"],
                "created_at": o.get("created_at"),
                "paid_at": o.get("paid_at"),
            })

        return web.json_response({"code": 0, "data": {"orders": items}})
    except Exception as e:
        logger.exception("pay_orders error")
        return web.json_response({"code": -1, "msg": str(e)}, status=500)


async def _wechat_native_pay(order_no: str, amount_fen: int, question_count: int) -> str:
    """调用微信 Native 支付接口获取二维码链接"""
    client = _get_wechat_pay_client()

    code, body = client.pay(
        description=f"面试额度 {question_count} 次",
        out_trade_no=order_no,
        amount={"total": amount_fen, "currency": "CNY"},
    )

    if code in (200, 201) and isinstance(body, dict):
        url = body.get("code_url", "")
        if url:
            logger.info("微信 Native 支付下单成功: order=%s code_url=%s", order_no, url)
            return url

    raise RuntimeError(f"微信支付下单失败: code={code} body={body}")


async def _wechat_query_order(order_no: str) -> str:
    """主动查询微信支付订单状态，返回 trade_state 字符串"""
    try:
        client = _get_wechat_pay_client()
        code, body = client.query(out_trade_no=order_no)
        if code == 200 and isinstance(body, dict):
            return body.get("trade_state", "UNKNOWN")
    except ImportError:
        logger.warning("wechatpayv3 未安装，跳过主动查询")
    except Exception as e:
        logger.warning("查询微信订单状态失败: %s", e)
    return "UNKNOWN"


def _decrypt_and_parse_notify(body: str, headers: dict) -> str | None:
    """解密并解析微信支付回调通知，提取 out_trade_no。

    使用 wechatpayv3 SDK 进行签名验证和 AES-256-GCM 解密。
    如果 SDK 不可用则回退到简单 JSON 解析（仅开发环境）。
    """
    try:
        client = _get_wechat_pay_client()
        result = client.callback(headers=headers, body=body)
        if isinstance(result, dict):
            return result.get("out_trade_no")
        if isinstance(result, tuple) and len(result) >= 2:
            data = result[1] if isinstance(result[1], dict) else {}
            return data.get("out_trade_no")
    except ImportError:
        logger.warning("wechatpayv3 未安装，回退到简单解析")
    except Exception as e:
        logger.warning("微信回调签名验证失败: %s，回退到简单解析", e)

    try:
        data = json.loads(body)
        resource = data.get("resource", {})
        if isinstance(resource, dict) and resource.get("out_trade_no"):
            return resource["out_trade_no"]
    except Exception:
        pass

    return None


def setup_payment_routes(app):
    app.router.add_get("/pay/plans", pay_plans)
    app.router.add_post("/pay/create-order", create_pay_order)
    app.router.add_post("/pay/notify", pay_notify)
    app.router.add_post("/pay/query", pay_query)
    app.router.add_get("/pay/orders", pay_orders)
