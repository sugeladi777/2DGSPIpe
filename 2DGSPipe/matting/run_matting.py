import argparse
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "runtime"))

import cv2
import numpy as np
from tqdm import tqdm
from soft_foreground_segmenter import SoftForegroundSegmenter


def is_image_file(name: str) -> bool:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
    return name.lower().endswith(exts)


def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(
        description="Demo script for depth estimation, foreground segmentation, and surface normal estimation"
    )
    parser.add_argument("--input_root", required=True, help="Path to input image")
    parser.add_argument(
        "--foreground-model", help="Path to foreground segmentation ONNX model", 
        default="model/foreground-segmentation-model-vitl16_384.onnx"
    )

    parser.add_argument("--output_root", required=True, help="Save result to a path")

    args = parser.parse_args()
    input_root = os.path.abspath(args.input_root)
    output_root = os.path.abspath(args.output_root)

    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"Input directory not found: {input_root}")

    os.makedirs(output_root, exist_ok=True)
    foreground_segmenter = SoftForegroundSegmenter(onnx_model=args.foreground_model)

    img_files = [
        name
        for name in sorted(os.listdir(input_root))
        if os.path.isfile(os.path.join(input_root, name)) and is_image_file(name)
    ]
    if not img_files:
        raise RuntimeError(f"No image files found in: {input_root}")

    invalid_count = 0
    for pth in tqdm(img_files):
        img_path = os.path.join(input_root, pth)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            invalid_count += 1
            continue

        fg = foreground_segmenter.estimate_foreground_segmentation(image)
        cv2.imwrite(
            os.path.join(output_root, pth), 
            (fg[..., None] * 255.).astype(np.uint8)
        )
    if invalid_count > 0:
        print(f"[Warning] skipped {invalid_count} unreadable images")


if __name__ == "__main__":
    main()
