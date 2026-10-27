"""Settings for the GitHub ADVI/VI usage pipeline: database location, output
directory, and the GitHub token."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    root: Path
    db_path: Path
    data_dir: Path
    github_token: str | None
    user_agent: str


def settings(db_path: Path | None = None) -> Settings:
    load_dotenv()
    default_db_path = ROOT / "data" / "github-vi-software.sqlite"
    resolved_db_path = db_path or Path(os.getenv("GITHUB_ADVI_DB", default_db_path))
    if not resolved_db_path.is_absolute():
        resolved_db_path = ROOT / resolved_db_path
    data_dir = (
        ROOT / "data"
        if resolved_db_path == default_db_path
        else resolved_db_path.parent / resolved_db_path.stem
    )
    return Settings(
        root=ROOT,
        db_path=resolved_db_path,
        data_dir=data_dir,
        github_token=os.getenv("GITHUB_TOKEN") or None,
        user_agent=os.getenv("GITHUB_ADVI_USER_AGENT", "github-advi-usage/0.1"),
    )
