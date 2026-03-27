import argparse
import os
import subprocess
import sys
from typing import Iterable, List, Optional


DEFAULT_GS_ITERATIONS = 8000
DEFAULT_GS_MESH_RES = 768


def run_cmd(cmd: Iterable[str], cwd: Optional[str] = None) -> None:
    subprocess.run(list(cmd), cwd=cwd, check=True)


def clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def build_gs_train_cmd(
    python_bin: str,
    data_root: str,
    recon_root: str,
    port: int,
    iterations: int,
) -> List[str]:
    # Keep the original training recipe, but compress it to a faster profile
    # that is more suitable for downstream mesh + UV + texture generation.
    densify_from_iter = clamp_int(iterations // 20, 200, 500)
    densify_until_iter = clamp_int(int(iterations * 0.75), densify_from_iter + 500, iterations)
    opacity_reset_interval = clamp_int(iterations // 4, 1000, 3000)

    return [
        python_bin,
        "train.py",
        "-s",
        data_root,
        "-m",
        recon_root,
        "--port",
        str(port),
        "--iterations",
        str(iterations),
        "--position_lr_max_steps",
        str(iterations),
        "--densify_from_iter",
        str(densify_from_iter),
        "--densify_until_iter",
        str(densify_until_iter),
        "--opacity_reset_interval",
        str(opacity_reset_interval),
        "--test_iterations",
        "-1",
        "--save_iterations",
        str(iterations),
        "--quiet",
    ]


def build_gs_render_cmd(
    python_bin: str,
    data_root: str,
    recon_root: str,
    mesh_res: int,
) -> List[str]:
    return [
        python_bin,
        "render.py",
        "-s",
        data_root,
        "-m",
        recon_root,
        "--skip_train",
        "--skip_test",
        "--mesh_res",
        str(mesh_res),
        "--quiet",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument(
        "--gs_iterations",
        type=int,
        default=DEFAULT_GS_ITERATIONS,
        help="2DGS 训练迭代数；默认使用更适合流水线的轻量配置",
    )
    parser.add_argument(
        "--gs_mesh_res",
        type=int,
        default=DEFAULT_GS_MESH_RES,
        help="2DGS mesh 导出分辨率；更低更快，但细节会略少",
    )
    opt = parser.parse_args()

    if opt.gs_iterations < 1000:
        raise ValueError("--gs_iterations 不能小于 1000")
    if opt.gs_mesh_res < 256:
        raise ValueError("--gs_mesh_res 不能小于 256")

    data_root = os.path.abspath(opt.data_root)
    code_root = os.path.dirname(os.path.abspath(__file__))
    python_bin = sys.executable

    run_cmd([python_bin, "to_2dgs_format.py", "--data_root", data_root], cwd=code_root)
    run_cmd([python_bin, "run_colmap.py", "--data_root", data_root], cwd=code_root)

    gs_code_root = os.path.join(code_root, "2d-gaussian-splatting")
    recon_root = os.path.join(data_root, "recon")
    port = int(os.environ.get("RECON_PORT", "6009"))

    run_cmd(
        build_gs_train_cmd(
            python_bin=python_bin,
            data_root=data_root,
            recon_root=recon_root,
            port=port,
            iterations=opt.gs_iterations,
        ),
        cwd=gs_code_root,
    )
    run_cmd(
        build_gs_render_cmd(
            python_bin=python_bin,
            data_root=data_root,
            recon_root=recon_root,
            mesh_res=opt.gs_mesh_res,
        ),
        cwd=gs_code_root,
    )

    run_cmd([python_bin, "to_my_format.py", "--data_root", data_root], cwd=code_root)


if __name__ == "__main__":
    main()
