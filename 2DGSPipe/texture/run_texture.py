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

    code_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.abspath(opt.data_root)
    python_bin = sys.executable

    run_cmd([python_bin, "render_gbuffer.py", "--data_root", data_root], cwd=code_root)
    run_cmd([python_bin, "build_texture.py", "--data_root", data_root], cwd=code_root)


if __name__ == "__main__":
    main()
