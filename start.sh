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

TRANSPORT="${TRANSPORT:-webrtc}"
HUMAN_MODEL="${HUMAN_MODEL:-wav2lip}"
AVATAR_ID="${AVATAR_ID:-wav2lip_avatar_veo1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-nerfstream}"
LISTEN_PORT="${LISTEN_PORT:-5001}"

APP_ARGS=(
  --transport "$TRANSPORT"
  --model "$HUMAN_MODEL"
  --avatar_id "$AVATAR_ID"
)

if [[ "$#" -gt 0 ]]; then
  APP_ARGS+=("$@")
fi

PORT_TO_CHECK="$LISTEN_PORT"
for ((i = 0; i < ${#APP_ARGS[@]}; i++)); do
  if [[ "${APP_ARGS[i]}" == "--listenport" ]] && (( i + 1 < ${#APP_ARGS[@]} )); then
    PORT_TO_CHECK="${APP_ARGS[i+1]}"
  fi
done

PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

if [[ -n "$PYTHON_BIN" ]]; then
  STARTUP_CHECK="$("$PYTHON_BIN" - "$PORT_TO_CHECK" <<'PY'
import socket
import sys
import urllib.error
import urllib.request

port = int(sys.argv[1])
sock = socket.socket()
sock.settimeout(1.5)
try:
    if sock.connect_ex(("127.0.0.1", port)) == 0:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/interview.html", timeout=2) as response:
                body = response.read(512).decode("utf-8", errors="ignore")
            if response.status == 200 and (
                "AI 数字人智能面试" in body
                or "开始面试" in body
                or "继续下一轮" in body
            ):
                print("already_running")
            else:
                print("port_busy")
        except Exception:
            print("port_busy")
    else:
        print("port_free")
except OSError:
    print("port_free")
finally:
    sock.close()
PY
)"

  if [[ "$STARTUP_CHECK" == "already_running" ]]; then
    echo "乔治面试 已在端口 $PORT_TO_CHECK 运行，无需重复启动。"
    echo "如需重启，请先执行: bash stop.sh"
    echo "或改用其他 LISTEN_PORT / --listenport。"
    exit 0
  fi

  if [[ "$STARTUP_CHECK" == "port_busy" ]]; then
    echo "端口 $PORT_TO_CHECK 已被其他进程占用，当前启动已取消。"
    echo "如果是当前项目实例，请先执行: bash stop.sh"
    echo "否则请释放端口，或改用其他 LISTEN_PORT / --listenport。"
    exit 1
  fi
fi

echo "Starting 乔治面试..."
echo "  transport: $TRANSPORT"
echo "  model: $HUMAN_MODEL"
echo "  avatar_id: $AVATAR_ID"

if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV_NAME" ]]; then
  exec python app.py "${APP_ARGS[@]}"
fi

if command -v conda >/dev/null 2>&1; then
  exec conda run --no-capture-output -n "$CONDA_ENV_NAME" python app.py "${APP_ARGS[@]}"
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 app.py "${APP_ARGS[@]}"
fi

exec python app.py "${APP_ARGS[@]}"
