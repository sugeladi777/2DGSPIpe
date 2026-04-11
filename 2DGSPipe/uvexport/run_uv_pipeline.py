import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Iterable, Optional

DEFAULT_BLENDER5_BIN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "blender-5.0.1-linux-x64",
    "blender",
)


def run_cmd(cmd: Iterable[str], cwd: Optional[str] = None) -> None:
    subprocess.run(list(cmd), cwd=cwd, check=True)


def is_image_file(name: str) -> bool:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
    return name.lower().endswith(exts)


def list_raw_image_names(raw_frames_root: str) -> list[str]:
    if not os.path.isdir(raw_frames_root):
        return []
    return sorted(
        [
            name
            for name in os.listdir(raw_frames_root)
            if os.path.isfile(os.path.join(raw_frames_root, name)) and is_image_file(name)
        ]
    )


def _to_float(value, default: float = -1e12) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _pick_existing_name(
    item: dict,
    index: int,
    available_names: list[str],
    available_set: set[str],
    stem_to_name: dict[str, str],
) -> str:
    # 1) Prefer backend-saved filename.
    for key in ("saved_filename", "filename"):
        name = str(item.get(key) or "").strip()
        if not name:
            continue
        base = os.path.basename(name)
        if base in available_set:
            return base
        # stem fallback: allow ext mismatch, e.g. xx.jpg vs xx.png
        stem = os.path.splitext(base)[0]
        if stem and stem in stem_to_name:
            return stem_to_name[stem]

    # 2) Legacy manifest fallback:
    # captured list order matches upload order; server saves as zero-padded sequence.
    if 0 <= index < len(available_names):
        return available_names[index]
    return ""


def pick_best_per_cell_from_manifest(
    manifest_path: str,
    available_names: list[str],
) -> tuple[set[str], int, int]:
    """
    Return (selected_file_names, selected_cell_count, unresolved_count).
    Selection policy: for each capture cell key, keep the highest-score frame.
    """
    if not os.path.isfile(manifest_path):
        return set(), 0, 0
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"[Frame Selection][Warning] failed to parse capture manifest: {e}")
        return set(), 0, 0

    captured = payload.get("captured") if isinstance(payload, dict) else None
    if not isinstance(captured, list) or not captured:
        return set(), 0, 0

    best_by_cell: dict[str, tuple[float, str]] = {}
    unresolved = 0
    available_set = set(available_names)
    stem_to_name = {
        os.path.splitext(name)[0]: name
        for name in available_names
    }
    for idx, item in enumerate(captured):
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        filename = _pick_existing_name(item, idx, available_names, available_set, stem_to_name)
        if not key or not filename:
            if key:
                unresolved += 1
            continue
        score = _to_float(item.get("score"), default=-1e12)
        cur = best_by_cell.get(key)
        if cur is None or score > cur[0]:
            best_by_cell[key] = (score, os.path.basename(filename))

    selected = {v[1] for v in best_by_cell.values() if v and v[1]}
    return selected, len(best_by_cell), unresolved


def sync_selected_raw_frames(
    raw_frames_root: str,
    image_root: str,
    selected_names: set[str],
) -> int:
    if not os.path.isdir(raw_frames_root):
        raise FileNotFoundError(f"Raw frame directory not found: {raw_frames_root}")
    os.makedirs(image_root, exist_ok=True)

    all_src_names = list_raw_image_names(raw_frames_root)
    if not all_src_names:
        raise RuntimeError(f"No image files found in raw frames: {raw_frames_root}")

    wanted = {os.path.basename(n) for n in selected_names if n}
    if not wanted:
        raise RuntimeError("Selected frame set is empty; one-per-cell selection requires non-empty manifest mapping.")
    src_names = [n for n in all_src_names if n in wanted]
    if not src_names:
        raise RuntimeError("Manifest selection produced 0 valid files in raw_frames.")
    src_name_set = set(src_names)

    for existing_name in os.listdir(image_root):
        existing_path = os.path.join(image_root, existing_name)
        if os.path.isfile(existing_path) and existing_name not in src_name_set:
            os.remove(existing_path)

    copied = 0
    for name in src_names:
        src_path = os.path.join(raw_frames_root, name)
        dst_path = os.path.join(image_root, name)
        should_copy = True
        if os.path.isfile(dst_path):
            src_stat = os.stat(src_path)
            dst_stat = os.stat(dst_path)
            should_copy = (dst_stat.st_size != src_stat.st_size) or (dst_stat.st_mtime < src_stat.st_mtime)
        if should_copy:
            shutil.copy2(src_path, dst_path)
        copied += 1
    print(
        f"[Frame Selection] mode=best_per_cell, selected={len(src_names)}, total_raw={len(all_src_names)}"
    )
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument(
        "--blender_bin",
        type=str,
        default=os.environ.get("BLENDER5_BIN", DEFAULT_BLENDER5_BIN),
        help="Blender 5 executable path",
    )
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
    blender_bin = os.path.abspath(opt.blender_bin)
    if not os.path.isfile(blender_bin):
        raise FileNotFoundError(f"Blender 5 binary not found: {blender_bin}")

    export_script = os.path.join(code_root, "export_uv_blender.py")
    run_cmd(
        [
            blender_bin,
            "--background",
            "--python",
            export_script,
            "--",
            input_mesh,
            uv_mesh,
        ],
        cwd=code_root,
    )
    if not os.path.isfile(uv_mesh):
        raise RuntimeError(f"UV export failed: output mesh not generated at {uv_mesh}")

    raw_frames_root = os.path.join(data_root, "raw_frames")
    texture_image_root = os.path.join(texture_dataset_root, "image")
    capture_manifest_path = os.path.join(data_root, "capture_manifest.json")
    raw_image_names = list_raw_image_names(raw_frames_root)
    selected_names, selected_cell_count, unresolved_count = pick_best_per_cell_from_manifest(
        capture_manifest_path,
        available_names=raw_image_names,
    )
    if not selected_names:
        raise RuntimeError(
            "No valid per-cell frame selected from capture_manifest.json. "
            "Current pipeline requires one best frame per existing cell."
        )
    print(
        f"[Frame Selection] manifest cells={selected_cell_count}, pick_one_per_cell={len(selected_names)}"
    )
    if unresolved_count:
        print(
            f"[Frame Selection][Warning] unresolved manifest entries={unresolved_count}; "
            "used available-file fallback for unmatched names."
        )
    copied_count = sync_selected_raw_frames(
        raw_frames_root=raw_frames_root,
        image_root=texture_image_root,
        selected_names=selected_names,
    )
    print(f"[Frame Selection] synced={copied_count}")

    run_cmd([python_bin, "prepare_data.py", "--data_root", data_root], cwd=code_root)


if __name__ == "__main__":
    main()
