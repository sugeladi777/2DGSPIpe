import argparse
import datetime
import os
import subprocess
import sys
from typing import Iterable, Optional


SUPPORTED_MODULES = ("mat", "face", "colmap", "2dgs", "uv", "tex")


def parse_modules(func_arg: str) -> list[str]:
    items = [x.strip() for x in func_arg.replace(",", "-").split("-") if x.strip()]
    if not items:
        raise ValueError("`--func` 不能为空")

    invalid: list[str] = []
    deduped: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key in SUPPORTED_MODULES:
            if key not in deduped:
                deduped.append(key)
            continue

        invalid.append(item)

    if invalid:
        raise ValueError(
            "不支持的模块: {}；可选模块: {}".format(
                sorted(set(invalid)),
                list(SUPPORTED_MODULES),
            )
        )

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


def _is_int_csv(text: str) -> bool:
    if not text:
        return False
    items = [x.strip() for x in text.split(",")]
    if not items:
        return False
    return all(x.isdigit() for x in items)


def _pick_best_gpu_index() -> Optional[str]:
    """
    Pick one GPU by a simple load score:
      score = memory_used_mb + 20 * utilization_percent
    Lower score is preferred.
    """
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        return None

    if res.returncode != 0 or not res.stdout:
        return None

    best_idx = None
    best_score = None
    for raw in res.stdout.splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 3:
            continue
        idx, mem_s, util_s = parts[0], parts[1], parts[2]
        if not idx.isdigit():
            continue
        try:
            mem = float(mem_s)
            util = float(util_s)
        except Exception:
            continue
        score = mem + 20.0 * util
        if best_score is None or score < best_score:
            best_score = score
            best_idx = idx
    return best_idx


def resolve_gpu_policy(gpu_policy: str, env: dict) -> tuple[Optional[str], str]:
    """
    Return:
      (cuda_visible_devices_value_or_none, description)
    """
    policy = (gpu_policy or "auto").strip().lower()
    existing = (env.get("CUDA_VISIBLE_DEVICES") or "").strip()

    if policy == "inherit":
        if existing:
            return existing, f"inherit CUDA_VISIBLE_DEVICES={existing}"
        return None, "inherit CUDA_VISIBLE_DEVICES (未设置)"

    if policy == "auto":
        if existing:
            return existing, f"auto 检测到已有 CUDA_VISIBLE_DEVICES={existing}，直接使用"
        picked = _pick_best_gpu_index()
        if picked is None:
            return None, "auto 选卡失败（nvidia-smi 不可用），沿用系统默认设备"
        return picked, f"auto 选择 GPU={picked}"

    if policy == "cpu":
        return "", "强制 CPU（CUDA_VISIBLE_DEVICES=''）"

    if _is_int_csv(policy):
        return policy, f"手动指定 GPU={policy}"

    raise ValueError("`--gpu` 仅支持 auto|inherit|cpu|<gpu_id[,gpu_id...]>")


def _resize_raw_frames_max_side(raw_root: str, max_side: int, log_path: str) -> None:
    if max_side <= 0:
        write_log(log_path, f"[Preprocess] skip resize: max_side={max_side}")
        return
    if not os.path.isdir(raw_root):
        write_log(log_path, f"[Preprocess] skip resize: raw frame dir not found: {raw_root}")
        return

    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError(f"需要 opencv-python 来缩放 raw_frames，但导入失败: {e}") from e

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
    names = [
        n
        for n in sorted(os.listdir(raw_root))
        if os.path.isfile(os.path.join(raw_root, n)) and n.lower().endswith(exts)
    ]
    if not names:
        write_log(log_path, f"[Preprocess] skip resize: no images in {raw_root}")
        return

    resized = 0
    skipped = 0
    for name in names:
        path = os.path.join(raw_root, name)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]
        long_side = max(h, w)
        if long_side <= max_side:
            skipped += 1
            continue

        scale = max_side / float(long_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ext = os.path.splitext(name)[1].lower()
        if ext in (".jpg", ".jpeg"):
            ok = cv2.imwrite(path, resized_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        elif ext == ".png":
            ok = cv2.imwrite(path, resized_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        else:
            ok = cv2.imwrite(path, resized_img)
        if not ok:
            raise RuntimeError(f"缩放后写入失败: {path}")
        resized += 1

    write_log(
        log_path,
        f"[Preprocess] resize raw_frames to max_side={max_side}: resized={resized}, unchanged_or_invalid={skipped}, total={len(names)}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--save_root", type=str, required=True, help="结果保存根目录")
    parser.add_argument("--func", type=str, default="mat-face-colmap-2dgs-uv-tex", help="执行模块")
    parser.add_argument(
        "--gpu",
        type=str,
        default="auto",
        help="GPU 策略: auto|inherit|cpu|<gpu_id[,gpu_id...]>",
    )
    parser.add_argument(
        "--max_image_side",
        type=int,
        default=1280,
        help="raw_frames 预处理: 长边最大值（短边等比缩放）",
    )
    opt = parser.parse_args()

    modules = parse_modules(opt.func)

    save_root = os.path.abspath(opt.save_root)
    os.makedirs(save_root, exist_ok=True)
    log_path = os.path.join(save_root, "log.txt")

    start_all = datetime.datetime.now()
    write_log(log_path, f"Start Job: {modules}")
    write_log(log_path, f"Start Time: {start_all.strftime('%Y-%m-%d %H:%M:%S')}")
    write_log(log_path, "-" * 30)

    code_root = os.path.dirname(os.path.abspath(__file__))
    python_bin = sys.executable
    run_env = os.environ.copy()
    cuda_visible_devices, gpu_note = resolve_gpu_policy(opt.gpu, run_env)
    if cuda_visible_devices is not None:
        run_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    write_log(log_path, f"[GPU] policy={opt.gpu}; {gpu_note}")

    raw_frame_root = os.path.join(save_root, "raw_frames")
    mask_save_root = os.path.join(save_root, "wholebody_mask")
    face_mask_save_root = os.path.join(save_root, "face_mask")
    tex_data_root = os.path.join(save_root, "texture_dataset")
    _resize_raw_frames_max_side(raw_frame_root, int(opt.max_image_side), log_path)

    module_specs = {
        "mat": {
            "cwd": os.path.join(code_root, "matting"),
            "cmd": [
                python_bin,
                "run_matting.py",
                "--input_root",
                raw_frame_root,
                "--output_root",
                mask_save_root,
            ],
        },
        "face": {
            "cwd": os.path.join(code_root, "face_mask"),
            "cmd": [
                python_bin,
                "run_face_detection.py",
                "--input_root",
                raw_frame_root,
                "--output_root",
                face_mask_save_root,
                "--batch_size",
                "12",
                "--det_max_size",
                "960",
            ],
        },
        "colmap": {
            "cwd": os.path.join(code_root, "reconstruction"),
            "cmd": [
                python_bin,
                "run_reconstruction.py",
                "--data_root",
                save_root,
                "--stage",
                "colmap",
            ],
        },
        "2dgs": {
            "cwd": os.path.join(code_root, "reconstruction"),
            "cmd": [
                python_bin,
                "run_reconstruction.py",
                "--data_root",
                save_root,
                "--stage",
                "2dgs",
            ],
        },
        "uv": {
            "cwd": os.path.join(code_root, "uvexport"),
            "cmd": [python_bin, "run_uv_pipeline.py", "--data_root", save_root],
        },
        "tex": {
            "cwd": os.path.join(code_root, "texture"),
            "cmd": [python_bin, "run_texture.py", "--data_root", tex_data_root],
        },
    }

    try:
        for module_name in modules:
            spec = module_specs[module_name]
            run_step(
                module_name=module_name,
                log_path=log_path,
                cwd=spec["cwd"],
                cmd=spec["cmd"],
                env=run_env,
            )
    finally:
        end_all = datetime.datetime.now()
        write_log(log_path, "-" * 30)
        write_log(log_path, f"total runtime: {end_all - start_all}")
        write_log(log_path, f"end time: {end_all.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
