import os
import subprocess
import mimetypes
import argparse
from multiprocessing import Pool
import sys


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


def run_job(args):
    dataset, video, gpu, port, func = args
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
    try:
        subprocess.run(cmd, env=env, check=True)
        return dataset, True, ""
    except subprocess.CalledProcessError as e:
        return dataset, False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/home/lichengkai/RGB_Recon/input/test")
    parser.add_argument("--output_root", type=str, default="/home/lichengkai/RGB_Recon/output/2026.3.27/less_fracture")
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated GPU IDs, e.g. 0,1")
    parser.add_argument("--base_port", type=int, default=6009)
    parser.add_argument("--func", type=str, default="extract-mat-face-recon-uv-tex")
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
        jobs.append((dataset, video, gpu, port, args.func))

    with Pool(processes=len(gpus)) as pool:
        results = pool.map(run_job, jobs)

    failed = [r for r in results if not r[1]]
    print(f"Total jobs: {len(results)}, Success: {len(results) - len(failed)}, Failed: {len(failed)}")
    for dataset, ok, info in failed:
        print(f"[FAILED] {dataset}: {info}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
