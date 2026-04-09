import os
import subprocess
import mimetypes
import argparse
from multiprocessing import Pool
import sys


def parse_modules(func_arg):
    supported = ["extract", "mat", "face", "recon", "uv", "tex"]
    items = [x.strip() for x in func_arg.replace(",", "-").split("-") if x.strip()]
    if not items:
        raise ValueError("`--func` 不能为空")
    invalid = sorted(set(items) - set(supported))
    if invalid:
        raise ValueError(f"不支持的模块: {invalid}；可选模块: {supported}")
    deduped = []
    for item in items:
        if item not in deduped:
            deduped.append(item)
    return deduped


def is_video_file(filename: str) -> bool:
    """Return True if filename looks like a video (MIME-based check with extension fallback)."""
    mime, _ = mimetypes.guess_type(filename)
    if mime and mime.startswith("video"):
        return True
    VIDEO_EXTS = (
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".flv",
        ".mpeg",
        ".mpg",
        ".m4v",
        ".wmv",
        ".3gp",
        ".ts",
        ".m2ts",
        ".mts",
    )
    return filename.lower().endswith(VIDEO_EXTS)


def is_job_complete(save_root, func):
    log_path = os.path.join(save_root, "log.txt")
    if not os.path.isfile(log_path):
        return False
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if "end time:" not in text:
        return False
    for module_name in parse_modules(func):
        if f"[Module: {module_name}] runtime:" not in text:
            return False
    return True


def run_job(args):
    dataset, video, gpu, port, func, candidate_step_size = args
    if is_job_complete(dataset, func):
        return dataset, True, "skipped"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["RECON_PORT"] = str(port)
    cmd = [
        sys.executable,
        "/home/lichengkai/RGB_Recon/2DGSPipe/run.py",
        "--save_root",
        dataset,
        "--video_path",
        video,
        "--func",
        func,
    ]
    if candidate_step_size > 0:
        cmd.extend(["--candidate_step_size", str(candidate_step_size)])
    try:
        subprocess.run(cmd, env=env, check=True)
        return dataset, True, ""
    except subprocess.CalledProcessError as e:
        return dataset, False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/home/lichengkai/RGB_Recon/input/test")
    parser.add_argument("--output_root", type=str, default="/home/lichengkai/RGB_Recon/output/2026.4.3/extrac_improve")
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated GPU IDs, e.g. 0,1")
    parser.add_argument("--base_port", type=int, default=6009)
    parser.add_argument("--func", type=str, default="extract-mat-face-recon-uv-tex")
    parser.add_argument("--candidate_step_size", type=int, default=10)
    args = parser.parse_args()

    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("`--gpus` 不能为空")

    input_dir = os.path.abspath(args.input_dir)
    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    video_paths = sorted(
        [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and is_video_file(f)
        ]
    )
    if not video_paths:
        print(f"No videos found under: {input_dir}")
        return

    jobs = []
    for i, video in enumerate(video_paths):
        stem = os.path.splitext(os.path.basename(video))[0]
        dataset = os.path.join(output_root, stem)
        gpu = gpus[i % len(gpus)]
        port = args.base_port + i
        jobs.append(
            (
                dataset,
                video,
                gpu,
                port,
                args.func,
                args.candidate_step_size,
            )
        )

    with Pool(processes=len(gpus)) as pool:
        results = pool.map(run_job, jobs)

    failed = [r for r in results if not r[1]]
    skipped = sum(1 for _, ok, info in results if ok and info == "skipped")
    print(
        f"Total jobs: {len(results)}, Success: {len(results) - len(failed)}, "
        f"Skipped: {skipped}, Failed: {len(failed)}"
    )
    for dataset, ok, info in failed:
        print(f"[FAILED] {dataset}: {info}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
