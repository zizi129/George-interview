import json
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

                CREATE TABLE IF NOT EXISTS resumes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER UNIQUE NOT NULL,
                    file_name TEXT DEFAULT '',
                    profile_json TEXT DEFAULT '{}',
                    job_title TEXT DEFAULT '',
                    updated_at REAL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS interview_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    job_title TEXT DEFAULT '',
                    interview_mode TEXT DEFAULT 'text',
                    score INTEGER DEFAULT 0,
                    recommendation TEXT DEFAULT '',
                    report_json TEXT DEFAULT '{}',
                    created_at REAL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
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


# ── 简历持久化 ────────────────────────────────────────────────────────────────

def save_user_resume(user_id: int, file_name: str, profile_dict: dict, job_title: str = "") -> bool:
    now = time.time()
    profile_text = json.dumps(profile_dict, ensure_ascii=False)
    with _db_lock:
        conn = _get_conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO resumes (user_id, file_name, profile_json, job_title, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, file_name or "", profile_text, job_title or "", now),
            )
            conn.commit()
            return True
        finally:
            conn.close()


def get_user_resume(user_id: int) -> dict | None:
    with _db_lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM resumes WHERE user_id = ?", (user_id,)
            ).fetchone()
            if not row:
                return None
            result = dict(row)
            try:
                result["profile"] = json.loads(result.pop("profile_json", "{}"))
            except (json.JSONDecodeError, TypeError):
                result["profile"] = {}
            return result
        finally:
            conn.close()


# ── 面试报告持久化 ─────────────────────────────────────────────────────────────

MAX_REPORTS_PER_USER = 5


def save_interview_report(
    user_id: int,
    job_title: str,
    interview_mode: str,
    score: int,
    recommendation: str,
    report_dict: dict,
) -> int:
    """保存面试报告并清理超出限额的旧记录，返回新记录 id。"""
    now = time.time()
    report_text = json.dumps(report_dict, ensure_ascii=False)
    with _db_lock:
        conn = _get_conn()
        try:
            cur = conn.execute(
                "INSERT INTO interview_reports "
                "(user_id, job_title, interview_mode, score, recommendation, report_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, job_title or "", interview_mode or "text",
                 score, recommendation or "", report_text, now),
            )
            new_id = cur.lastrowid
            conn.execute(
                "DELETE FROM interview_reports WHERE user_id = ? AND id NOT IN "
                "(SELECT id FROM interview_reports WHERE user_id = ? ORDER BY created_at DESC LIMIT ?)",
                (user_id, user_id, MAX_REPORTS_PER_USER),
            )
            conn.commit()
            return new_id
        finally:
            conn.close()


def get_user_interview_reports(user_id: int, limit: int = MAX_REPORTS_PER_USER) -> list[dict]:
    with _db_lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM interview_reports WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
            results = []
            for row in rows:
                r = dict(row)
                try:
                    r["report"] = json.loads(r.pop("report_json", "{}"))
                except (json.JSONDecodeError, TypeError):
                    r["report"] = {}
                results.append(r)
            return results
        finally:
            conn.close()
