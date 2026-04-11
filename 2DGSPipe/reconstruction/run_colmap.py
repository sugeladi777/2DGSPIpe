import argparse
import json
import os
import shutil
import subprocess
from typing import Iterable


DEFAULT_CAMERA_MODEL = "PINHOLE"
DEFAULT_FEATURE_TYPE = "ALIKED_N16ROT"
DEFAULT_MATCHER_TYPE = "ALIKED_LIGHTGLUE"

SUPPORTED_FEATURE_TYPES = ("ALIKED_N16ROT", "ALIKED_N32")
SUPPORTED_MATCHER_TYPES = (
    "ALIKED_BRUTEFORCE",
    "ALIKED_LIGHTGLUE",
)


def run_cmd(cmd: Iterable[str]) -> None:
    subprocess.run(list(cmd), check=True)


def get_required_local_colmap_bin() -> str:
    local_build_bin = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "colmap",
        "build",
        "src",
        "colmap",
        "exe",
        "colmap",
    )
    if os.path.isfile(local_build_bin) and os.access(local_build_bin, os.X_OK):
        return local_build_bin
    raise FileNotFoundError(
        f"Required COLMAP binary not found or not executable: {local_build_bin}. "
        "Please build COLMAP at this location before running the pipeline."
    )


def colmap_cmd(colmap_bin: str, *args: str) -> list[str]:
    return [colmap_bin, *args]


def is_image_file(name: str) -> bool:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
    return name.lower().endswith(exts)


def remove_path(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)


def count_input_images(img_root: str) -> int:
    if not os.path.isdir(img_root):
        return 0
    return sum(
        1
        for n in os.listdir(img_root)
        if os.path.isfile(os.path.join(img_root, n)) and is_image_file(n)
    )


def prepare_colmap_mask_root(img_root: str, src_mask_root: str) -> str:
    """
    COLMAP expects mask path format:
      <mask_root>/<relative_image_name>.png
    For image '00001.png', expected mask name is '00001.png.png'.
    This function only validates expected masks and does not create aliases.
    """
    if not os.path.isdir(src_mask_root):
        return ""
    if not os.path.isdir(img_root):
        return ""

    total_images = 0
    prepared = 0
    missing = 0

    for name in sorted(os.listdir(img_root)):
        img_path = os.path.join(img_root, name)
        if not os.path.isfile(img_path) or not is_image_file(name):
            continue
        total_images += 1

        expected_mask_name = f"{name}.png"
        expected_mask_path = os.path.join(src_mask_root, expected_mask_name)
        if os.path.isfile(expected_mask_path):
            prepared += 1
        else:
            missing += 1

    if total_images <= 0:
        return ""
    if prepared <= 0 or missing > 0:
        print(
            f"[COLMAP][Warning] masks incomplete: prepared={prepared}, "
            f"missing={missing}, total_images={total_images}. Disable mask for COLMAP."
        )
        return ""

    print(f"[COLMAP] using mask root: prepared={prepared}, root={src_mask_root}")
    return src_mask_root


def prepare_mask_root_with_mode(data_root: str, img_root: str, colmap_mask_mode: str) -> str:
    mode = (colmap_mask_mode or "auto").strip().lower()
    if mode == "face":
        primary = "face_mask"
    elif mode == "wholebody":
        primary = "wholebody_mask"
    else:
        primary = "wholebody_mask"
        manifest_path = os.path.join(data_root, "capture_manifest.json")
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    capture_mode = str(
                        payload.get("capture_mode")
                        or payload.get("mode")
                        or payload.get("shoot_mode")
                        or ""
                    ).strip().lower().replace("-", "_").replace(" ", "_")
                    if capture_mode in {
                        "head_rotate",
                        "rotate_head",
                        "head_rotation",
                        "stationary_phone",
                        "phone_static",
                    }:
                        primary = "face_mask"
            except Exception:
                pass

    primary_root = os.path.join(data_root, primary)
    print(f"[COLMAP] mask mode={colmap_mask_mode}, selected={primary}")
    mask_root = prepare_colmap_mask_root(img_root, primary_root)
    if mask_root:
        return mask_root

    # If face mask is selected but unavailable, fallback to wholebody mask.
    if primary == "face_mask":
        fallback_root = os.path.join(data_root, "wholebody_mask")
        print("[COLMAP][Warning] face_mask unavailable, fallback to wholebody_mask")
        return prepare_colmap_mask_root(img_root, fallback_root)
    return ""


def resolve_matcher_type(feature_type: str, matcher_type: str) -> str:
    mtype = (matcher_type or DEFAULT_MATCHER_TYPE).strip().upper()
    ftype = feature_type.strip().upper()
    if not ftype.startswith("ALIKED"):
        raise ValueError("当前流水线仅支持 ALIKED 特征")
    if not mtype.startswith("ALIKED_"):
        raise ValueError("当前流水线仅支持 ALIKED 匹配器")
    return mtype


def build_feature_extractor_cmd(
    colmap_bin: str,
    database_root: str,
    img_root: str,
    mask_root: str,
    camera_model: str,
    sift_max_image_size: int,
    feature_type: str,
    single_camera: int,
    aliked_max_num_features: int,
) -> list[str]:
    ftype = feature_type.strip().upper()
    cmd = [
        colmap_bin,
        "feature_extractor",
        "--database_path",
        database_root,
        "--image_path",
        img_root,
        "--ImageReader.camera_model",
        camera_model,
        "--ImageReader.single_camera",
        str(int(single_camera)),
        "--FeatureExtraction.type",
        ftype,
        "--FeatureExtraction.use_gpu",
        "1",
        "--FeatureExtraction.max_image_size",
        str(sift_max_image_size),
    ]

    if ftype.startswith("ALIKED"):
        cmd.extend(["--AlikedExtraction.max_num_features", str(aliked_max_num_features)])
    else:
        raise ValueError("当前流水线仅支持 ALIKED 特征")

    if os.path.isdir(mask_root):
        cmd.extend(["--ImageReader.mask_path", mask_root])
    return cmd


def build_matcher_cmd(
    colmap_bin: str,
    database_root: str,
    matcher_type: str,
) -> list[str]:
    guided_matching = "0" if "LIGHTGLUE" in matcher_type.upper() else "1"
    return [
        colmap_bin,
        "exhaustive_matcher",
        "--database_path",
        database_root,
        "--FeatureMatching.type",
        matcher_type,
        "--FeatureMatching.guided_matching",
        guided_matching,
        "--FeatureMatching.use_gpu",
        "1",
        "--TwoViewGeometry.filter_stationary_matches",
        "1",
    ]


def build_mapper_cmd(colmap_bin: str, database_root: str, img_root: str, sparse_root: str) -> list[str]:
    return [
        colmap_bin,
        "mapper",
        "--database_path",
        database_root,
        "--image_path",
        img_root,
        "--output_path",
        sparse_root,
        "--Mapper.tri_ignore_two_view_tracks",
        "0",
    ]


def model_subdirs(sparse_root: str) -> list[str]:
    if not os.path.isdir(sparse_root):
        return []
    out = []
    for name in os.listdir(sparse_root):
        p = os.path.join(sparse_root, name)
        if os.path.isdir(p) and name.isdigit():
            out.append(p)
    return sorted(out, key=lambda p: int(os.path.basename(p)))


def ensure_model_txt(colmap_bin: str, model_dir: str) -> None:
    run_cmd(
        colmap_cmd(
            colmap_bin,
            "model_converter",
            "--input_path",
            model_dir,
            "--output_path",
            model_dir,
            "--output_type",
            "TXT",
        )
    )


def count_registered_images_in_model(model_dir: str) -> int:
    images_txt = os.path.join(model_dir, "images.txt")
    if not os.path.isfile(images_txt):
        return 0
    non_comment_lines = []
    with open(images_txt, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            non_comment_lines.append(line)
    return len(non_comment_lines) // 2


def run_colmap_pipeline(
    *,
    colmap_bin: str,
    data_root: str,
    img_root: str,
    mask_root: str,
    camera_model: str,
    sift_max_image_size: int,
    feature_type: str,
    matcher_type: str,
    single_camera: int,
    aliked_max_num_features: int,
) -> tuple[str, str, int]:
    database_root = os.path.join(data_root, "database.db")
    sparse_root = os.path.join(data_root, "sparse")
    remove_path(database_root)
    remove_path(sparse_root)
    os.makedirs(sparse_root, exist_ok=True)

    run_cmd(
        build_feature_extractor_cmd(
            colmap_bin=colmap_bin,
            database_root=database_root,
            img_root=img_root,
            mask_root=mask_root,
            camera_model=camera_model,
            sift_max_image_size=sift_max_image_size,
            feature_type=feature_type,
            single_camera=single_camera,
            aliked_max_num_features=aliked_max_num_features,
        )
    )
    run_cmd(
        build_matcher_cmd(
            colmap_bin=colmap_bin,
            database_root=database_root,
            matcher_type=matcher_type,
        )
    )
    run_cmd(
        build_mapper_cmd(
            colmap_bin=colmap_bin,
            database_root=database_root,
            img_root=img_root,
            sparse_root=sparse_root,
        )
    )

    best_model_dir = None
    best_registered = -1
    for model_dir in model_subdirs(sparse_root):
        ensure_model_txt(colmap_bin, model_dir)
        reg_count = count_registered_images_in_model(model_dir)
        if reg_count > best_registered:
            best_registered = reg_count
            best_model_dir = model_dir

    if not best_model_dir:
        raise RuntimeError("COLMAP failed: no valid reconstruction model produced.")

    run_cmd(
        colmap_cmd(
            colmap_bin,
            "bundle_adjuster",
            "--input_path",
            best_model_dir,
            "--output_path",
            best_model_dir,
            "--BundleAdjustmentCeres.max_num_iterations",
            "100",
        )
    )
    ensure_model_txt(colmap_bin, best_model_dir)

    return (
        database_root,
        best_model_dir,
        count_registered_images_in_model(best_model_dir),
    )


def finalize_best_result(
    colmap_bin: str,
    data_root: str,
    database_path: str,
    model_dir: str,
) -> None:
    final_sparse_root = os.path.join(data_root, "sparse")
    final_recon_root = os.path.join(final_sparse_root, "0")
    final_database_root = os.path.join(data_root, "database.db")

    src_model_dir = os.path.abspath(model_dir)
    dst_model_dir = os.path.abspath(final_recon_root)
    src_database = os.path.abspath(database_path)
    dst_database = os.path.abspath(final_database_root)

    if src_model_dir != dst_model_dir:
        # Keep source model intact even when it lives under sparse/.
        # Only replace sparse/0 instead of deleting the whole sparse root.
        remove_path(final_recon_root)
        os.makedirs(final_sparse_root, exist_ok=True)
        shutil.copytree(model_dir, final_recon_root)

    if src_database != dst_database:
        shutil.copy2(database_path, final_database_root)

    ensure_model_txt(colmap_bin, final_recon_root)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument(
        "--colmap_mask_mode",
        type=str,
        default="auto",
        choices=["auto", "face", "wholebody"],
    )
    parser.add_argument("--sift_max_image_size", type=int, default=4000)
    parser.add_argument("--camera_model", type=str, default=DEFAULT_CAMERA_MODEL)
    parser.add_argument("--single_camera", type=int, default=1, choices=[0, 1])
    parser.add_argument(
        "--feature_type",
        type=str,
        default=DEFAULT_FEATURE_TYPE,
        choices=SUPPORTED_FEATURE_TYPES,
        help="COLMAP 特征类型",
    )
    parser.add_argument(
        "--matcher_type",
        type=str,
        default=DEFAULT_MATCHER_TYPE,
        choices=SUPPORTED_MATCHER_TYPES,
        help="COLMAP 匹配器类型（仅 ALIKED）",
    )
    parser.add_argument(
        "--aliked_max_num_features",
        type=int,
        default=2048,
        help="ALIKED 的每图最大特征数",
    )
    args = parser.parse_args()
    data_root = os.path.abspath(args.data_root)
    colmap_bin = get_required_local_colmap_bin()
    print(f"[COLMAP] binary={colmap_bin}")

    img_root = os.path.join(data_root, "images")
    mask_root = prepare_mask_root_with_mode(data_root, img_root, args.colmap_mask_mode)
    feature_type = args.feature_type.strip().upper()
    matcher_type = resolve_matcher_type(feature_type, args.matcher_type)
    camera_model = args.camera_model
    total_input_images = count_input_images(img_root)
    if total_input_images <= 0:
        raise RuntimeError(f"No images found in: {img_root}")

    database_path, model_dir, registered_images = run_colmap_pipeline(
        colmap_bin=colmap_bin,
        data_root=data_root,
        img_root=img_root,
        mask_root=mask_root,
        camera_model=camera_model,
        sift_max_image_size=args.sift_max_image_size,
        feature_type=feature_type,
        matcher_type=matcher_type,
        single_camera=args.single_camera,
        aliked_max_num_features=args.aliked_max_num_features,
    )
    finalize_best_result(colmap_bin, data_root, database_path, model_dir)
    best_ratio = registered_images / float(total_input_images)
    print(
        f"[COLMAP] matcher=exhaustive/{matcher_type}, feature={feature_type}, "
        f"registered={registered_images}/{total_input_images} ({best_ratio * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
