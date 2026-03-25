from __future__ import annotations

import os
from pathlib import Path


def load_env_file(env_path: str | None = None) -> Path:
    base_dir = Path(__file__).resolve().parent
    path = Path(env_path) if env_path else base_dir / ".env"

    if not path.is_file():
        return path

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        if value == "":
            continue

        os.environ.setdefault(key, value)

    return path
