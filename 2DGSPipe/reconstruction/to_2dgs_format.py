import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm


def is_image_file(name: str) -> bool:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
    return name.lower().endswith(exts)


def build_rgba_image(img_path: str, mask_path: str, use_mask: bool) -> np.ndarray:
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    h, w = img_bgr.shape[:2]
    alpha = np.full((h, w), 255, dtype=np.uint8)

    if use_mask and os.path.isfile(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            alpha = mask

    img_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    img_bgra[..., 3] = alpha
    return img_bgra


def process_one(
    name: str,
    img_root: str,
    mask_root: str,
    save_root: str,
    use_mask: bool,
    png_compression: int,
    skip_existing: bool,
) -> str:
    save_path = os.path.join(save_root, name)
    if skip_existing and os.path.isfile(save_path):
        return "skip"

    img_path = os.path.join(img_root, name)
    mask_path = os.path.join(mask_root, name)
    rgba = build_rgba_image(img_path, mask_path, use_mask)
    ok = cv2.imwrite(save_path, rgba, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
    if not ok:
        raise RuntimeError(f"Failed to write image: {save_path}")
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument(
        "--mask_root",
        type=str,
        default=None,
        help="Mask folder path. Defaults to <data_root>/face_mask",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Thread workers for parallel image processing",
    )
    parser.add_argument(
        "--png_compression",
        type=int,
        default=1,
        help="PNG compression level [0-9], smaller is faster",
    )
    parser.add_argument(
        "--skip_existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Skip images already written in output folder (default: on)",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Force overwrite existing images",
    )
    opt = parser.parse_args()

    data_root = os.path.abspath(opt.data_root)
    img_root = os.path.join(data_root, "raw_frames")
    mask_root = os.path.abspath(opt.mask_root) if opt.mask_root else os.path.join(data_root, "face_mask")
    save_root = os.path.join(data_root, "images")
    os.makedirs(save_root, exist_ok=True)

    if not os.path.isdir(img_root):
        raise FileNotFoundError(f"Input image directory not found: {img_root}")

    use_mask = os.path.isdir(mask_root)
    if not use_mask:
        print(f"[Warning] mask directory not found, fallback to full foreground mask: {mask_root}")

    image_files = [
        name
        for name in sorted(os.listdir(img_root))
        if os.path.isfile(os.path.join(img_root, name)) and is_image_file(name)
    ]
    if not image_files:
        raise RuntimeError(f"No image files found in: {img_root}")

    if not 0 <= opt.png_compression <= 9:
        raise ValueError("--png_compression must be in [0, 9]")
    workers = max(1, opt.workers)

    ok_count = 0
    skip_count = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_one,
                name,
                img_root,
                mask_root,
                save_root,
                use_mask,
                opt.png_compression,
                opt.skip_existing,
            )
            for name in image_files
        ]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            result = fut.result()
            if result == "ok":
                ok_count += 1
            elif result == "skip":
                skip_count += 1

    print(f"[to_2dgs_format] done: wrote={ok_count}, skipped={skip_count}, total={len(image_files)}")


if __name__ == "__main__":
    main()
