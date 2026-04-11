import argparse
import os
import subprocess
import sys
from typing import Iterable, Optional


def run_cmd(cmd: Iterable[str], cwd: Optional[str] = None) -> None:
    subprocess.run(list(cmd), cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--persistent_workers", type=int, default=1)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    opt = parser.parse_args()

    code_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.abspath(opt.data_root)
    python_bin = sys.executable

    worker_args = [
        "--num_workers",
        str(max(0, int(opt.num_workers))),
        "--persistent_workers",
        "1" if int(opt.persistent_workers) else "0",
        "--prefetch_factor",
        str(max(1, int(opt.prefetch_factor))),
    ]
    run_cmd([python_bin, "render_gbuffer.py", "--data_root", data_root, *worker_args], cwd=code_root)
    run_cmd([python_bin, "build_texture.py", "--data_root", data_root, *worker_args], cwd=code_root)


if __name__ == "__main__":
    main()
