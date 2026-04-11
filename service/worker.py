import hashlib
import json
import os
import subprocess
from pathlib import Path

from service.api.config import settings
from service.api.db import get_job, update_job, utc_now_iso


def _parse_modules(func: str) -> list[str]:
    supported = ["mat", "face", "recon", "colmap", "2dgs", "uv", "tex"]
    items = [x.strip().lower() for x in func.replace(",", "-").split("-") if x.strip()]
    expanded: list[str] = []
    for item in items:
        if item == "recon":
            expanded.extend(["colmap", "2dgs"])
        else:
            expanded.append(item)

    deduped = []
    for item in expanded:
        if item in supported and item not in deduped:
            deduped.append(item)
    return deduped


def _has_images(root: Path) -> bool:
    if not root.is_dir():
        return False
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    for name in root.iterdir():
        if name.is_file() and name.suffix.lower() in exts:
            return True
    return False


def _pick_recon_port(job_id: str) -> int:
    digest = hashlib.md5(job_id.encode("utf-8")).hexdigest()
    slot = int(digest[:6], 16) % 500
    return settings.recon_port_base + slot


def _collect_artifacts(save_root: Path) -> list[dict]:
    candidates = [
        "log.txt",
        "worker.log",
        "mesh/2dgs_recon.obj",
        "texture_dataset/final_textured.obj",
        "texture_dataset/final_hack_texture.png",
        "texture_dataset/final_hack.obj",
        "texture_dataset/final_hack.mtl",
    ]
    artifacts = []
    for rel in candidates:
        p = save_root / rel
        if p.is_file():
            artifacts.append(
                {
                    "path": rel,
                    "size": p.stat().st_size,
                }
            )
    return artifacts


def run_pipeline_job(job_id: str) -> None:
    job = get_job(job_id)
    if not job:
        raise RuntimeError(f"job not found: {job_id}")

    save_root = Path(job["save_root"]).resolve()
    save_root.mkdir(parents=True, exist_ok=True)
    worker_log_path = save_root / "worker.log"

    update_job(
        job_id,
        status="running",
        started_at=utc_now_iso(),
        error=None,
    )

    modules = _parse_modules(str(job["func"]))
    if not modules:
        update_job(
            job_id,
            status="failed",
            ended_at=utc_now_iso(),
            error=f"无有效模块可执行: {job.get('func')}",
            pipeline_pid=None,
        )
        return
    normalized_func = "-".join(modules)
    cmd = [
        settings.pipeline_python,
        str(settings.pipeline_entry),
        "--save_root",
        str(save_root),
        "--func",
        normalized_func,
        "--gpu",
        "auto",
        "--max_image_side",
        "1280",
    ]
    raw_frames_root = save_root / "raw_frames"
    if not _has_images(raw_frames_root):
        update_job(
            job_id,
            status="failed",
            ended_at=utc_now_iso(),
            error=f"未检测到输入图片，请检查目录: {raw_frames_root}",
            pipeline_pid=None,
        )
        return

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["RECON_PORT"] = str(_pick_recon_port(job_id))
    # Force pipeline-side automatic GPU selection in 2DGSPipe/run.py.
    # Remove inherited / configured CUDA pinning so --gpu auto can really pick.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if settings.blender_bin:
        env["BLENDER5_BIN"] = settings.blender_bin

    with worker_log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[worker] cmd: {' '.join(cmd)}\n")
        log_file.write("[worker] gpu policy: force --gpu auto (unset CUDA_VISIBLE_DEVICES)\n")
        log_file.write(f"[worker] start: {utc_now_iso()}\n")
        log_file.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=settings.repo_root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        update_job(job_id, pipeline_pid=proc.pid)
        ret = proc.wait()

        log_file.write(f"[worker] end: {utc_now_iso()}, return_code={ret}\n")
        log_file.flush()

    if ret != 0:
        update_job(
            job_id,
            status="failed",
            ended_at=utc_now_iso(),
            error=f"pipeline exited with code {ret}",
            pipeline_pid=None,
        )
        return

    artifacts = _collect_artifacts(save_root)
    manifest_path = save_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    update_job(
        job_id,
        status="succeeded",
        ended_at=utc_now_iso(),
        error=None,
        pipeline_pid=None,
    )
