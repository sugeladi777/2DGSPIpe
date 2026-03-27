import argparse
import datetime
import os
import subprocess
import sys
from typing import Iterable, List, Optional


def parse_modules(func_arg: str) -> List[str]:
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


def write_log(log_path: str, message: str) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")
    print(message, flush=True)


def run_step(
    *,
    module_name: str,
    cmd: Iterable[str],
    log_path: str,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
) -> None:
    m_start = datetime.datetime.now()
    write_log(log_path, f"[Module: {module_name}] start")
    cmd_list = list(cmd)
    write_log(log_path, f"[Module: {module_name}] cmd: {' '.join(cmd_list)}")
    subprocess.run(cmd_list, cwd=cwd, env=env, check=True)
    m_end = datetime.datetime.now()
    write_log(log_path, f"[Module: {module_name}] runtime: {m_end - m_start}")


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video_path", type=str, help="输入视频路径")
    parser.add_argument("--video_step_size", default=10, type=int, help="帧提取间隔（每隔 N 帧取一帧）")
    parser.add_argument("--video_ds_ratio", default=0.5, type=float, help="视频下采样比例")
    parser.add_argument("--reg_close_eye", type=int, default=0, help="是否使用闭眼模板（当前版本未启用）")
    parser.add_argument("--save_root", type=str, required=True, help="结果保存根目录")
    parser.add_argument("--func", type=str, default="extract-mat-face-recon-uv-tex", help="执行模块")
    opt = parser.parse_args()

    modules = parse_modules(opt.func)
    if "extract" in modules and not opt.video_path:
        parser.error("当包含 extract 模块时，必须提供 --video_path")

    save_root = os.path.abspath(opt.save_root)
    os.makedirs(save_root, exist_ok=True)
    log_path = os.path.join(save_root, "log.txt")

    start_all = datetime.datetime.now()
    write_log(log_path, f"Start Job: {modules}")
    write_log(log_path, f"Start Time: {start_all.strftime('%Y-%m-%d %H:%M:%S')}")
    write_log(log_path, "-" * 30)

    code_root = os.path.dirname(os.path.abspath(__file__))
    python_bin = sys.executable

    raw_frame_root = os.path.join(save_root, "raw_frames")
    mask_save_root = os.path.join(save_root, "wholebody_mask")
    face_mask_save_root = os.path.join(save_root, "face_mask")

    try:
        if "extract" in modules:
            os.makedirs(raw_frame_root, exist_ok=True)
            output_pattern = os.path.join(raw_frame_root, "%05d.png")
            vf_expr = (
                f"select=not(mod(n\\,{opt.video_step_size})),"
                f"scale=iw*{opt.video_ds_ratio}:ih*{opt.video_ds_ratio},setsar=1:1"
            )
            run_step(
                module_name="extract",
                log_path=log_path,
                cmd=[
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    opt.video_path,
                    "-vf",
                    vf_expr,
                    "-vsync",
                    "vfr",
                    "-q:v",
                    "1",
                    output_pattern,
                ],
            )

        if "mat" in modules:
            run_step(
                module_name="mat",
                log_path=log_path,
                cwd=os.path.join(code_root, "matting"),
                cmd=[
                    python_bin,
                    "run_matting.py",
                    "--input_root",
                    raw_frame_root,
                    "--output_root",
                    mask_save_root,
                ],
            )

        if "face" in modules:
            run_step(
                module_name="face",
                log_path=log_path,
                cwd=os.path.join(code_root, "face_mask"),
                cmd=[
                    python_bin,
                    "run_face_detection.py",
                    "--input_root",
                    raw_frame_root,
                    "--output_root",
                    face_mask_save_root,
                ],
            )

        if "recon" in modules:
            run_step(
                module_name="recon",
                log_path=log_path,
                cwd=os.path.join(code_root, "reconstruction"),
                cmd=[python_bin, "run_reconstruction.py", "--data_root", save_root],
            )

        if "uv" in modules:
            run_step(
                module_name="uv",
                log_path=log_path,
                cwd=os.path.join(code_root, "uvexport"),
                cmd=[python_bin, "run_uv_pipeline.py", "--data_root", save_root],
            )

        if "tex" in modules:
            tex_data_root = os.path.join(save_root, "texture_dataset")
            run_step(
                module_name="tex",
                log_path=log_path,
                cwd=os.path.join(code_root, "texture"),
                cmd=[python_bin, "run_texture.py", "--data_root", tex_data_root],
            )
    finally:
        end_all = datetime.datetime.now()
        write_log(log_path, "-" * 30)
        write_log(log_path, f"total runtime: {end_all - start_all}")
        write_log(log_path, f"end time: {end_all.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
