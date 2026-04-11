import os
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE_ROOT = Path(__file__).resolve().parents[1]


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return int(raw)


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return float(raw)


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    service_root: Path
    web_root: Path
    jobs_root: Path
    db_path: Path

    redis_url: str
    queue_name: str
    job_timeout_sec: int

    pipeline_entry: Path
    pipeline_python: str
    pipeline_func: str
    pipeline_candidate_step_size: int
    pipeline_video_ds_ratio: float
    pipeline_video_max_side: int

    blender_bin: str | None
    recon_port_base: int



def load_settings() -> Settings:
    jobs_root = Path(os.environ.get("JOBS_ROOT", str(REPO_ROOT / "service_data" / "jobs")))
    db_path = Path(os.environ.get("DB_PATH", str(jobs_root / "jobs.db")))

    pipeline_entry = Path(
        os.environ.get(
            "PIPELINE_ENTRY",
            str(REPO_ROOT / "2DGSPipe" / "run.py"),
        )
    )

    return Settings(
        repo_root=REPO_ROOT,
        service_root=SERVICE_ROOT,
        web_root=SERVICE_ROOT / "web",
        jobs_root=jobs_root,
        db_path=db_path,
        redis_url=os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0"),
        queue_name=os.environ.get("RQ_QUEUE_NAME", "rgb_recon"),
        job_timeout_sec=_env_int("JOB_TIMEOUT_SEC", 6 * 60 * 60),
        pipeline_entry=pipeline_entry,
        pipeline_python=os.environ.get("PIPELINE_PYTHON", sys.executable),
        pipeline_func=os.environ.get("PIPELINE_FUNC", "mat-face-colmap-2dgs-uv-tex"),
        pipeline_candidate_step_size=_env_int("PIPELINE_CANDIDATE_STEP_SIZE", 10),
        pipeline_video_ds_ratio=_env_float("PIPELINE_VIDEO_DS_RATIO", 0.5),
        pipeline_video_max_side=_env_int("PIPELINE_VIDEO_MAX_SIDE", 1280),
        blender_bin=os.environ.get("BLENDER5_BIN"),
        recon_port_base=_env_int("RECON_PORT_BASE", 6009),
    )


settings = load_settings()
