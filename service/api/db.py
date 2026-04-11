import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from service.api.config import settings


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_conn():
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                video_filename TEXT NOT NULL,
                video_path TEXT NOT NULL,
                save_root TEXT NOT NULL,
                func TEXT NOT NULL,
                candidate_step_size INTEGER NOT NULL,
                video_ds_ratio REAL NOT NULL,
                video_max_side INTEGER NOT NULL,
                queue_job_id TEXT,
                pipeline_pid INTEGER,
                created_at TEXT NOT NULL,
                started_at TEXT,
                ended_at TEXT,
                error TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {k: row[k] for k in row.keys()}


def create_job(record: dict[str, Any]) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                id, status, video_filename, video_path, save_root,
                func, candidate_step_size, video_ds_ratio, video_max_side,
                queue_job_id, pipeline_pid,
                created_at, started_at, ended_at, error, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                record["status"],
                record["video_filename"],
                record["video_path"],
                record["save_root"],
                record["func"],
                record["candidate_step_size"],
                record["video_ds_ratio"],
                record["video_max_side"],
                record.get("queue_job_id"),
                record.get("pipeline_pid"),
                record["created_at"],
                record.get("started_at"),
                record.get("ended_at"),
                record.get("error"),
                record["updated_at"],
            ),
        )


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = utc_now_iso()
    keys = list(fields.keys())
    assigns = ", ".join([f"{k} = ?" for k in keys])
    values = [fields[k] for k in keys]
    values.append(job_id)

    with get_conn() as conn:
        conn.execute(f"UPDATE jobs SET {assigns} WHERE id = ?", values)


def get_job(job_id: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return row_to_dict(cur.fetchone())


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        )
        return [row_to_dict(row) for row in cur.fetchall()]
