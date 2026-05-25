import os
from pathlib import Path


def project_root_from(file_path: str) -> Path:
    current = Path(file_path).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "utils").exists() and (
            (candidate / "task").exists() or (candidate / "annotate").exists() or (candidate / "script").exists()
        ):
            return candidate
    return current.parent


def load_project_env(root: Path) -> Path:
    env_path = root / ".env"
    if not env_path.exists():
        return env_path

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        value = os.path.expanduser(os.path.expandvars(value))
        os.environ.setdefault(key, value)
    return env_path
