import cv2
import os
import argparse
from tqdm import tqdm
import torch


def is_image_file(name: str) -> bool:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
    return name.lower().endswith(exts)


def compute_sharpness(pth: str) -> float:
    """Estimate image sharpness. Bigger is better."""
    img = cv2.imread(pth, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    lap = cv2.Laplacian(img, cv2.CV_8UC1, ksize=3)
    return float(cv2.mean(lap)[0])


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--img_root", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    opt = parser.parse_args()

    img_root = os.path.abspath(opt.img_root)
    save_root = os.path.abspath(opt.save_root)
    if not os.path.isdir(img_root):
        raise FileNotFoundError(f"Image root not found: {img_root}")

    os.makedirs(save_root, exist_ok=True)
    shap_save_path = os.path.join(save_root, "sharpness.pkl")
    sharpness = {}
    image_files = [
        pth
        for pth in sorted(os.listdir(img_root))
        if os.path.isfile(os.path.join(img_root, pth)) and is_image_file(pth)
    ]
    if not image_files:
        raise RuntimeError(f"No image files found in: {img_root}")

    for pth in tqdm(image_files):
        sharpness[pth] = compute_sharpness(os.path.join(img_root, pth))
    torch.save(sharpness, shap_save_path)


if __name__ == "__main__":
    main()
