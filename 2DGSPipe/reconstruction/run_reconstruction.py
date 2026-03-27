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
    opt = parser.parse_args()

    data_root = os.path.abspath(opt.data_root)
    code_root = os.path.dirname(os.path.abspath(__file__))
    python_bin = sys.executable

    run_cmd([python_bin, "to_2dgs_format.py", "--data_root", data_root], cwd=code_root)
    run_cmd([python_bin, "run_colmap.py", "--data_root", data_root], cwd=code_root)

    gs_code_root = os.path.join(code_root, "2d-gaussian-splatting")
    recon_root = os.path.join(data_root, "recon")
    port = os.environ.get("RECON_PORT", "6009")
    run_cmd([python_bin, "train.py", "-s", data_root, "-m", recon_root, "--port", str(port)], cwd=gs_code_root)
    run_cmd([python_bin, "render.py", "-s", data_root, "-m", recon_root], cwd=gs_code_root)

    run_cmd([python_bin, "to_my_format.py", "--data_root", data_root], cwd=code_root)


if __name__ == "__main__":
    main()
