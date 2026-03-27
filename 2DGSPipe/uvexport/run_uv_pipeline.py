import argparse
import os
import subprocess
import sys
from typing import Iterable, Optional

BLENDER5_BIN = "/home/lichengkai/RGB_Recon/2DGSPipe/uvexport/blender-5.0.1-linux-x64/blender"


def run_cmd(cmd: Iterable[str], cwd: Optional[str] = None) -> None:
    subprocess.run(list(cmd), cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_view", type=int, default=16)
    opt = parser.parse_args()

    data_root = os.path.abspath(opt.data_root)
    code_root = os.path.dirname(os.path.abspath(__file__))
    python_bin = sys.executable

    input_mesh = os.path.join(data_root, "mesh", "2dgs_recon.obj")
    uv_mesh = os.path.join(data_root, "texture_dataset", "final_hack.obj")
    texture_dataset_root = os.path.dirname(uv_mesh)
    os.makedirs(texture_dataset_root, exist_ok=True)

    if not os.path.isfile(input_mesh):
        raise FileNotFoundError(f"Input mesh not found: {input_mesh}")
    if not os.path.isfile(BLENDER5_BIN):
        raise FileNotFoundError(f"Blender 5 binary not found: {BLENDER5_BIN}")

    run_cmd(
        [
            BLENDER5_BIN,
            "--background",
            "--python",
            os.path.join(code_root, "export_uv_blender.py"),
            "--",
            input_mesh,
            uv_mesh,
        ],
        cwd=code_root,
    )
    if not os.path.isfile(uv_mesh):
        raise RuntimeError(f"UV export failed: output mesh not generated at {uv_mesh}")

    raw_frames_root = os.path.join(data_root, "raw_frames")
    transforms_path = os.path.join(data_root, "mesh", "transforms.json")
    run_cmd(
        [
            python_bin,
            "select_frame/compute_sharpness.py",
            "--img_root",
            raw_frames_root,
            "--save_root",
            texture_dataset_root,
        ],
        cwd=code_root,
    )
    run_cmd(
        [
            python_bin,
            "select_frame/sample_by_sharpness.py",
            "--img_root",
            raw_frames_root,
            "--cam_path",
            transforms_path,
            "--save_root",
            texture_dataset_root,
            "--num_view",
            str(opt.num_view),
        ],
        cwd=code_root,
    )
    run_cmd([python_bin, "prepare_data.py", "--data_root", data_root], cwd=code_root)


if __name__ == "__main__":
    main()
