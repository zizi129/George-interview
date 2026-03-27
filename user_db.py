import sqlite3
import os
import time
import uuid
from threading import Lock

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "users.db")
_db_lock = Lock()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _db_lock:
        conn = _get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phone TEXT UNIQUE NOT NULL,
                    nickname TEXT DEFAULT '',
                    free_quota INTEGER DEFAULT 2,
                    paid_quota INTEGER DEFAULT 0,
                    created_at REAL,
                    last_login_at REAL
                );

                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    order_no TEXT UNIQUE NOT NULL,
                    question_count INTEGER NOT NULL,
                    amount_fen INTEGER NOT NULL,
                    status TEXT DEFAULT 'pending',
                    wechat_prepay_id TEXT,
                    created_at REAL,
                    paid_at REAL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
            """)
            conn.commit()
        finally:
            conn.close()


def get_or_create_user(phone: str) -> dict:
    now = time.time()
    with _db_lock:
        conn = _get_conn()
        try:
            row = conn.execute("SELECT * FROM users WHERE phone = ?", (phone,)).fetchone()
            if row:
                conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (now, row["id"]))
                conn.commit()
                row = conn.execute("SELECT * FROM users WHERE id = ?", (row["id"],)).fetchone()
                return dict(row)
            conn.execute(
                "INSERT INTO users (phone, free_quota, paid_quota, created_at, last_login_at) VALUES (?, 2, 0, ?, ?)",
                (phone, now, now),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM users WHERE phone = ?", (phone,)).fetchone()
            return dict(row)
        finally:
            conn.close()


def get_user_by_id(user_id: int) -> dict | None:
    with _db_lock:
        conn = _get_conn()
        try:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def get_user_quota(user_id: int) -> dict:
    user = get_user_by_id(user_id)
    if not user:
        return {"free_quota": 0, "paid_quota": 0, "total": 0}
    return {
        "free_quota": user["free_quota"],
        "paid_quota": user["paid_quota"],
        "total": user["free_quota"] + user["paid_quota"],
    }


def deduct_quota(user_id: int) -> bool:
    with _db_lock:
        conn = _get_conn()
        try:
            row = conn.execute("SELECT free_quota, paid_quota FROM users WHERE id = ?", (user_id,)).fetchone()
            if not row:
                return False
            if row["free_quota"] > 0:
                conn.execute("UPDATE users SET free_quota = free_quota - 1 WHERE id = ?", (user_id,))
            elif row["paid_quota"] > 0:
                conn.execute("UPDATE users SET paid_quota = paid_quota - 1 WHERE id = ?", (user_id,))
            else:
                return False
            conn.commit()
            return True
        finally:
            conn.close()


def add_paid_quota(user_id: int, count: int) -> bool:
    if count <= 0:
        return False
    with _db_lock:
        conn = _get_conn()
        try:
            row = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
            if not row:
                return False
            conn.execute("UPDATE users SET paid_quota = paid_quota + ? WHERE id = ?", (count, user_id))
            conn.commit()
            return True
        finally:
            conn.close()


def generate_order_no() -> str:
    ts = time.strftime("%Y%m%d%H%M%S")
    return f"ORD{ts}{uuid.uuid4().hex[:8].upper()}"


def create_order(user_id: int, question_count: int, amount_fen: int) -> dict:
    order_no = generate_order_no()
    now = time.time()
    with _db_lock:
        conn = _get_conn()
        try:
            conn.execute(
                "INSERT INTO orders (user_id, order_no, question_count, amount_fen, status, created_at) VALUES (?, ?, ?, ?, 'pending', ?)",
                (user_id, order_no, question_count, amount_fen, now),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM orders WHERE order_no = ?", (order_no,)).fetchone()
            return dict(row)
        finally:
            conn.close()


def get_order_by_no(order_no: str) -> dict | None:
    with _db_lock:
        conn = _get_conn()
        try:
            row = conn.execute("SELECT * FROM orders WHERE order_no = ?", (order_no,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def mark_order_paid(order_no: str, wechat_prepay_id: str = "") -> bool:
    now = time.time()
    with _db_lock:
        conn = _get_conn()
        try:
            row = conn.execute("SELECT * FROM orders WHERE order_no = ? AND status = 'pending'", (order_no,)).fetchone()
            if not row:
                return False
            conn.execute(
                "UPDATE orders SET status = 'paid', paid_at = ?, wechat_prepay_id = ? WHERE order_no = ?",
                (now, wechat_prepay_id, order_no),
            )
            conn.execute(
                "UPDATE users SET paid_quota = paid_quota + ? WHERE id = ?",
                (row["question_count"], row["user_id"]),
            )
            conn.commit()
            return True
        finally:
            conn.close()


def mark_order_failed(order_no: str) -> bool:
    with _db_lock:
        conn = _get_conn()
        try:
            conn.execute(
                "UPDATE orders SET status = 'failed' WHERE order_no = ? AND status = 'pending'",
                (order_no,),
            )
            conn.commit()
            return conn.total_changes > 0
        finally:
            conn.close()


def get_user_orders(user_id: int, limit: int = 20) -> list[dict]:
    with _db_lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM orders WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
