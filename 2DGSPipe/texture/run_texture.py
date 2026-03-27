import argparse
import json
import os
import subprocess
import sys
from typing import Iterable, Optional


def run_cmd(cmd: Iterable[str], cwd: Optional[str] = None) -> None:
    subprocess.run(list(cmd), cwd=cwd, check=True)


def load_expected_frame_stems(transforms_path: str):
    with open(transforms_path, "r") as f:
        meta = json.load(f)
    return [
        os.path.splitext(os.path.basename(frame["file_path"]))[0]
        for frame in meta.get("frames", [])
    ]


def render_outputs_ready(data_root: str, expected_stems):
    uv_root = os.path.join(data_root, "uv")
    uv_mask_root = os.path.join(data_root, "uv_mask")
    if not os.path.isdir(uv_root) or not os.path.isdir(uv_mask_root):
        return False
    for stem in expected_stems:
        if not os.path.isfile(os.path.join(uv_root, f"{stem}.pkl")):
            return False
        if not os.path.isfile(os.path.join(uv_mask_root, f"{stem}.png")):
            return False
    return True


def oldest_output_mtime(data_root: str, expected_stems):
    oldest = None
    for stem in expected_stems:
        for path in (
            os.path.join(data_root, "uv", f"{stem}.pkl"),
            os.path.join(data_root, "uv_mask", f"{stem}.png"),
        ):
            cur_mtime = os.path.getmtime(path)
            oldest = cur_mtime if oldest is None else min(oldest, cur_mtime)
    return 0.0 if oldest is None else oldest


def should_skip_render_gbuffer(data_root: str, code_root: str) -> bool:
    transforms_path = os.path.join(data_root, "transforms.json")
    mesh_path = os.path.join(data_root, "final_hack.obj")
    render_script = os.path.join(code_root, "render_gbuffer.py")
    if not os.path.isfile(transforms_path) or not os.path.isfile(mesh_path):
        return False
    expected_stems = load_expected_frame_stems(transforms_path)
    if not expected_stems or not render_outputs_ready(data_root, expected_stems):
        return False
    oldest_mtime = oldest_output_mtime(data_root, expected_stems)
    for dep in (transforms_path, mesh_path, render_script):
        if os.path.getmtime(dep) > oldest_mtime:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    opt = parser.parse_args()

    code_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.abspath(opt.data_root)
    python_bin = sys.executable

    if should_skip_render_gbuffer(data_root, code_root):
        print(f"[render_gbuffer] skip existing cached outputs under: {data_root}")
    else:
        run_cmd([python_bin, "render_gbuffer.py", "--data_root", data_root], cwd=code_root)
    run_cmd([python_bin, "build_texture.py", "--data_root", data_root], cwd=code_root)


if __name__ == "__main__":
    main()
