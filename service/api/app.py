import os
import json
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from service.api.config import settings
from service.api.db import create_job, get_job, init_db, list_jobs, update_job, utc_now_iso
from service.api.progress import MODULE_ORDER, parse_progress
from service.api.queueing import enqueue_job


app = FastAPI(title="RGB Recon Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    settings.jobs_root.mkdir(parents=True, exist_ok=True)
    init_db()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


def _normalize_modules(func: str) -> List[str]:
    supported = {"mat", "face", "recon", "colmap", "2dgs", "uv", "tex"}
    modules = [x.strip().lower() for x in func.replace(",", "-").split("-") if x.strip()]
    if not modules:
        raise HTTPException(status_code=400, detail="func 不能为空")
    invalid = sorted(set(modules) - supported)
    if invalid:
        raise HTTPException(status_code=400, detail=f"包含不支持模块: {invalid}")

    expanded: List[str] = []
    for m in modules:
        if m == "recon":
            expanded.extend(["colmap", "2dgs"])
        else:
            expanded.append(m)

    deduped = []
    for m in expanded:
        if m not in deduped:
            deduped.append(m)
    return deduped


def _sanitize_image_ext(name: str) -> str:
    ext = Path(name or "").suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}:
        return ext
    return ".jpg"


@app.post("/api/jobs/images")
async def create_recon_job_from_images(
    images: List[UploadFile] = File(...),
    func: str = Form(settings.pipeline_func),
    guide_meta_json: str = Form(""),
) -> dict:
    modules = _normalize_modules(func)
    normalized_func = "-".join(modules)
    if not images:
        raise HTTPException(status_code=400, detail="至少需要上传一张图片")

    job_id = uuid.uuid4().hex
    job_root = settings.jobs_root / job_id
    save_root = job_root / "work"
    raw_frames_root = save_root / "raw_frames"
    save_root.mkdir(parents=True, exist_ok=True)
    raw_frames_root.mkdir(parents=True, exist_ok=True)

    valid_count = 0
    saved_files = []
    for idx, image in enumerate(images, start=1):
        payload = await image.read()
        if not payload:
            continue
        ext = _sanitize_image_ext(image.filename or "")
        out_name = f"{idx:05d}{ext}"
        out_path = raw_frames_root / out_name
        out_path.write_bytes(payload)
        valid_count += 1
        saved_files.append(out_name)

    if valid_count == 0:
        raise HTTPException(status_code=400, detail="上传图片为空或不可读")

    if guide_meta_json.strip():
        manifest_payload = None
        try:
            manifest_payload = json.loads(guide_meta_json)
        except Exception:
            manifest_payload = {"raw_text": guide_meta_json}

        if isinstance(manifest_payload, dict):
            captured = manifest_payload.get("captured")
            if isinstance(captured, list):
                for i, item in enumerate(captured):
                    if not isinstance(item, dict):
                        continue
                    if i < len(saved_files):
                        item["saved_filename"] = saved_files[i]
            manifest_payload["saved_files"] = saved_files

        (save_root / "capture_manifest.json").write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    now = utc_now_iso()
    record = {
        "id": job_id,
        "status": "queued",
        "video_filename": "__images__",
        "video_path": "",
        "save_root": str(save_root),
        "func": normalized_func,
        "candidate_step_size": int(settings.pipeline_candidate_step_size),
        "video_ds_ratio": float(settings.pipeline_video_ds_ratio),
        "video_max_side": int(settings.pipeline_video_max_side),
        "queue_job_id": None,
        "pipeline_pid": None,
        "created_at": now,
        "started_at": None,
        "ended_at": None,
        "error": None,
        "updated_at": now,
    }
    create_job(record)

    try:
        queue_job_id = enqueue_job(job_id)
    except Exception as exc:
        update_job(job_id, status="failed", error=f"enqueue failed: {exc}")
        raise HTTPException(status_code=500, detail=f"enqueue failed: {exc}") from exc

    update_job(job_id, queue_job_id=queue_job_id)
    return {
        "job_id": job_id,
        "status": "queued",
        "queue_job_id": queue_job_id,
        "saved_images": saved_files,
    }


@app.get("/api/jobs")
def get_jobs(limit: int = Query(default=50, ge=1, le=200)) -> dict:
    jobs = list_jobs(limit=limit)
    return {"jobs": jobs}


@app.get("/api/jobs/{job_id}")
def get_job_detail(job_id: str) -> dict:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    func_tokens = [x.strip().lower() for x in job["func"].replace(",", "-").split("-") if x.strip()]
    if "recon" in func_tokens:
        func_tokens.extend(["colmap", "2dgs"])
    modules = [m for m in MODULE_ORDER if m in func_tokens]
    progress = parse_progress(Path(job["save_root"]) / "log.txt", modules)

    return {
        **job,
        "progress": progress,
    }


@app.get("/api/jobs/{job_id}/logs")
def get_job_logs(
    job_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=65536, ge=1, le=2_000_000),
) -> dict:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    log_path = Path(job["save_root"]) / "log.txt"
    if not log_path.is_file():
        return {
            "text": "",
            "offset": offset,
            "next_offset": offset,
            "eof": True,
        }

    file_size = log_path.stat().st_size
    if offset > file_size:
        offset = file_size

    with log_path.open("rb") as f:
        f.seek(offset)
        data = f.read(limit)

    next_offset = offset + len(data)
    return {
        "text": data.decode("utf-8", errors="ignore"),
        "offset": offset,
        "next_offset": next_offset,
        "eof": next_offset >= file_size,
    }


@app.get("/api/jobs/{job_id}/result")
def get_job_result(job_id: str) -> dict:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] != "succeeded":
        raise HTTPException(status_code=409, detail=f"job status is {job['status']}")

    save_root = Path(job["save_root"]).resolve()
    candidates = [
        "log.txt",
        "mesh/2dgs_recon.obj",
        "texture_dataset/final_textured.obj",
        "texture_dataset/final_hack_texture.png",
        "texture_dataset/final_hack.obj",
        "texture_dataset/final_hack.mtl",
        "manifest.json",
    ]

    artifacts = []
    for rel in candidates:
        p = save_root / rel
        if p.is_file():
            artifacts.append(
                {
                    "path": rel,
                    "size": p.stat().st_size,
                    "download_url": f"/api/jobs/{job_id}/artifacts/{rel}",
                }
            )

    return {
        "job_id": job_id,
        "artifacts": artifacts,
    }


@app.get("/api/jobs/{job_id}/artifacts/{artifact_path:path}")
def get_artifact(job_id: str, artifact_path: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    save_root = Path(job["save_root"]).resolve()
    target = (save_root / artifact_path).resolve()
    if os.path.commonpath([str(target), str(save_root)]) != str(save_root):
        raise HTTPException(status_code=400, detail="invalid artifact path")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")

    return FileResponse(target)


if settings.web_root.is_dir():
    app.mount("/", StaticFiles(directory=settings.web_root, html=True), name="web")
