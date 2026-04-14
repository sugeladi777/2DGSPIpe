#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _read_job_ids_file(path: str) -> list[str]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"job_ids_file not found: {path}")
    out: list[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def _normalize_save_root(path_str: str) -> str:
    p = Path(path_str).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"save_root path not found: {p}")

    # Accept both <job>/work and <job>.
    if (p / "raw_frames").is_dir():
        return str(p)
    work = p / "work"
    if work.is_dir() and (work / "raw_frames").is_dir():
        return str(work)

    raise RuntimeError(f"cannot resolve save_root(work) from: {p}")


def _collect_save_roots(repo_root: Path, save_roots: list[str], job_ids: list[str]) -> list[str]:
    roots: list[str] = []

    for root in save_roots:
        roots.append(_normalize_save_root(root))

    jobs_root = repo_root / "service_data" / "jobs"
    for job_id in job_ids:
        jid = str(job_id).strip()
        if not jid:
            continue
        roots.append(_normalize_save_root(str(jobs_root / jid)))

    # Deduplicate while preserving order.
    deduped: list[str] = []
    seen: set[str] = set()
    for r in roots:
        if r in seen:
            continue
        seen.add(r)
        deduped.append(r)
    return deduped


def _run_one(python_bin: str, run_py: str, save_root: str, func: str, gpu: str, max_image_side: int) -> None:
    cmd = [
        python_bin,
        run_py,
        "--save_root",
        save_root,
        "--func",
        func,
        "--gpu",
        gpu,
        "--max_image_side",
        str(int(max_image_side)),
    ]
    print(f"\n=== RUN: {save_root} ===", flush=True)
    print("CMD:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch rerun selected 2DGSPipe datasets by save_root/job_id.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--save_roots",
        nargs="*",
        default=[],
        help="One or more save_root paths. Supports both <job>/work and <job>.",
    )
    parser.add_argument(
        "--job_ids",
        nargs="*",
        default=[],
        help="Job IDs under service_data/jobs; each resolves to <repo>/service_data/jobs/<id>/work.",
    )
    parser.add_argument(
        "--job_ids_file",
        default="",
        help="Text file containing one job_id per line.",
    )
    parser.add_argument("--func", default="mat-face-colmap-2dgs-uv-tex", help="Modules passed to run.py --func")
    parser.add_argument("--gpu", default="auto", help="GPU policy for run.py")
    parser.add_argument("--max_image_side", type=int, default=1280)
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_py = str((repo_root / "2DGSPipe" / "run.py").resolve())
    python_bin = sys.executable

    job_ids = list(args.job_ids)
    if args.job_ids_file.strip():
        job_ids.extend(_read_job_ids_file(args.job_ids_file.strip()))

    save_roots = _collect_save_roots(repo_root, list(args.save_roots), job_ids)
    if not save_roots:
        raise RuntimeError("No datasets selected. Use --save_roots and/or --job_ids/--job_ids_file.")

    print("Selected datasets:")
    for i, root in enumerate(save_roots, start=1):
        print(f"  {i}. {root}")

    if args.dry_run:
        print("\nDry run only. Nothing executed.")
        return

    failures: list[tuple[str, str]] = []
    for root in save_roots:
        try:
            _run_one(
                python_bin=python_bin,
                run_py=run_py,
                save_root=root,
                func=args.func,
                gpu=args.gpu,
                max_image_side=args.max_image_side,
            )
        except Exception as e:
            failures.append((root, str(e)))
            print(f"[ERROR] {root}: {e}", flush=True)
            if not args.continue_on_error:
                break

    if failures:
        print("\nBatch finished with failures:")
        for root, err in failures:
            print(f"  - {root}: {err}")
        raise SystemExit(1)

    print("\nBatch finished successfully.")


if __name__ == "__main__":
    main()
