#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f ".env" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "${line//[[:space:]]/}" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue

    line="${line#export }"
    key="${line%%=*}"
    value="${line#*=}"

    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"

    [[ -z "$key" || -z "$value" ]] && continue

    if [[ "$value" == \"*\" && "$value" == *\" ]]; then
      value="${value:1:-1}"
    elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
      value="${value:1:-1}"
    fi

    export "$key=$value"
  done < ".env"
fi

PORT_TO_CHECK="${LISTEN_PORT:-5001}"

if [[ "${1:-}" == "--listenport" && -n "${2:-}" ]]; then
  PORT_TO_CHECK="$2"
elif [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  PORT_TO_CHECK="$1"
fi

PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "未找到 python3/python，无法停止 乔治面试 服务。"
  exit 1
fi

"$PYTHON_BIN" - "$PORT_TO_CHECK" <<'PY'
import os
import signal
import sys
import time

TARGET_PORT = int(sys.argv[1])
LISTEN_STATES = {"0A"}


def iter_listener_inodes(port: int) -> set[str]:
    inodes: set[str] = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(table, "r", encoding="utf-8") as fh:
                next(fh, None)
                for line in fh:
                    parts = line.split()
                    if len(parts) < 10:
                        continue
                    local_addr = parts[1]
                    state = parts[3]
                    inode = parts[9]
                    try:
                        _, port_hex = local_addr.split(":")
                    except ValueError:
                        continue
                    if int(port_hex, 16) == port and state in LISTEN_STATES:
                        inodes.add(inode)
        except FileNotFoundError:
            continue
    return inodes


def read_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\x00", b" ").decode().strip()
    except Exception:
        return ""


def find_listener_processes(inodes: set[str]) -> dict[int, str]:
    matches: dict[int, str] = {}
    for pid_text in os.listdir("/proc"):
        if not pid_text.isdigit():
            continue
        pid = int(pid_text)
        fd_dir = f"/proc/{pid}/fd"
        try:
            for fd in os.listdir(fd_dir):
                target = os.readlink(os.path.join(fd_dir, fd))
                if target.startswith("socket:[") and target[8:-1] in inodes:
                    matches[pid] = read_cmdline(pid)
                    break
        except Exception:
            continue
    return matches


def is_livetalking_process(cmd: str) -> bool:
    return (
        "python app.py" in cmd
        or "conda run --no-capture-output -n nerfstream python app.py" in cmd
        or "/workspace/LiveTalking/app.py" in cmd
    )


def wait_for_exit(pids: list[int], timeout_s: float) -> list[int]:
    deadline = time.time() + timeout_s
    alive = list(pids)
    while time.time() < deadline and alive:
        next_alive = []
        for pid in alive:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            except PermissionError:
                next_alive.append(pid)
            else:
                next_alive.append(pid)
        alive = next_alive
        if alive:
            time.sleep(0.2)
    return alive


listener_inodes = iter_listener_inodes(TARGET_PORT)
if not listener_inodes:
    print(f"端口 {TARGET_PORT} 上没有正在监听的 乔治面试 进程。")
    raise SystemExit(0)

listener_processes = find_listener_processes(listener_inodes)
livetalking_processes = {
    pid: cmd for pid, cmd in listener_processes.items() if is_livetalking_process(cmd)
}

if not livetalking_processes:
    print(f"端口 {TARGET_PORT} 正在被其他进程占用，stop.sh 不会强制停止它。")
    for pid, cmd in sorted(listener_processes.items()):
        print(f"  pid={pid} cmd={cmd or '<unknown>'}")
    raise SystemExit(1)

target_pids = sorted(livetalking_processes)
for pid in target_pids:
    cmd = livetalking_processes[pid] or "<unknown>"
    print(f"Stopping 乔治面试 pid={pid}")
    print(f"  cmd: {cmd}")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

alive = wait_for_exit(target_pids, timeout_s=6.0)
if alive:
    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    alive = wait_for_exit(alive, timeout_s=2.0)

if alive:
    print("以下进程仍未退出：")
    for pid in alive:
        print(f"  pid={pid}")
    raise SystemExit(1)

print(f"乔治面试 已停止，端口 {TARGET_PORT} 已释放。")
PY
