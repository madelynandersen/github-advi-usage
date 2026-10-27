"""SQLite schema and helpers for the GitHub ADVI/VI usage database."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY,
            stage TEXT NOT NULL,
            entity_id TEXT,
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            payload_json TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS github_repositories (
            full_name TEXT PRIMARY KEY,
            html_url TEXT NOT NULL,
            description TEXT,
            language TEXT,
            stars INTEGER,
            forks INTEGER,
            pushed_at TEXT,
            raw_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS github_files (
            file_id TEXT PRIMARY KEY,
            repo_full_name TEXT NOT NULL,
            path TEXT NOT NULL,
            html_url TEXT NOT NULL,
            raw_url TEXT,
            sha TEXT,
            raw_json TEXT NOT NULL,
            scan_status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(repo_full_name, path),
            FOREIGN KEY(repo_full_name) REFERENCES github_repositories(full_name)
        );

        CREATE TABLE IF NOT EXISTS github_advi_calls (
            id INTEGER PRIMARY KEY,
            file_id TEXT NOT NULL,
            repo_full_name TEXT NOT NULL,
            path TEXT NOT NULL,
            html_url TEXT NOT NULL,
            line_number INTEGER NOT NULL,
            package TEXT NOT NULL,
            call_kind TEXT NOT NULL,
            function_name TEXT NOT NULL,
            method TEXT,
            changed_parameters TEXT NOT NULL,
            keyword_arguments TEXT NOT NULL,
            positional_arguments TEXT NOT NULL,
            source_snippet TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(file_id, line_number, function_name, source_snippet),
            FOREIGN KEY(file_id) REFERENCES github_files(file_id)
        );

        CREATE TABLE IF NOT EXISTS github_advi_call_classifications (
            call_id INTEGER PRIMARY KEY,
            usage_status TEXT NOT NULL,
            is_false_positive INTEGER NOT NULL,
            category TEXT NOT NULL,
            confidence REAL NOT NULL,
            reason TEXT NOT NULL,
            classifier_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(call_id) REFERENCES github_advi_calls(id)
        );

        CREATE INDEX IF NOT EXISTS idx_github_calls_repo
            ON github_advi_calls(repo_full_name, package, call_kind);
        CREATE INDEX IF NOT EXISTS idx_github_files_status
            ON github_files(scan_status, repo_full_name);
        CREATE INDEX IF NOT EXISTS idx_github_call_classifications_status
            ON github_advi_call_classifications(usage_status, category);

        CREATE TABLE IF NOT EXISTS github_advi_parameter_overrides (
            repo_full_name TEXT NOT NULL,
            path TEXT NOT NULL,
            line_number INTEGER NOT NULL,
            override_optimizer_expression TEXT,
            override_loss_expression TEXT,
            override_sample_or_mc_expression TEXT,
            reason TEXT NOT NULL,
            source TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(repo_full_name, path, line_number)
        );

        CREATE TABLE IF NOT EXISTS github_advi_settings_companions (
            call_id INTEGER PRIMARY KEY,
            settings_call_id INTEGER,
            resolution_scope TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(call_id) REFERENCES github_advi_calls(id),
            FOREIGN KEY(settings_call_id) REFERENCES github_advi_calls(id)
        );
        """
    )
    conn.commit()


def log_event(
    conn: sqlite3.Connection,
    stage: str,
    message: str,
    *,
    entity_id: str | None = None,
    level: str = "info",
    payload: dict[str, Any] | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO events(stage, entity_id, level, message, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (stage, entity_id, level, message, json.dumps(payload or {}), utc_now()),
    )
    conn.commit()
