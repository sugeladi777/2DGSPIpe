#!/usr/bin/env python3
import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from batch_face import RetinaFace, FarlParser


def is_image_file(name: str) -> bool:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
    return name.lower().endswith(exts)


def main():
    parser = argparse.ArgumentParser(description="Generate soft face masks using batch-face")
    parser.add_argument("--input_root", required=True, help="Path to input images")
    parser.add_argument("--output_root", required=True, help="Path to save face masks (with hair)")
    parser.add_argument(
        "--output_root_no_hair",
        default="",
        help="Path to save face masks without hair; default is sibling folder '<output_root>_no_hair'",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--det_max_size", type=int, default=960)
    parser.add_argument("--det_threshold", type=float, default=0.6)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    output_root_no_hair = args.output_root_no_hair.strip()
    os.makedirs(output_root_no_hair, exist_ok=True)

    face_detector = RetinaFace(gpu_id=args.gpu_id, network="resnet50", return_dict=True)
    face_parser = FarlParser(gpu_id=args.gpu_id, name="farl/lapa/448")
    batch_size = args.batch_size

    input_files = sorted(
        f
        for f in os.listdir(args.input_root)
        if os.path.isfile(os.path.join(args.input_root, f)) and is_image_file(f)
    )
    for i in tqdm(range(0, len(input_files), batch_size)):
        batch_files = input_files[i : i + batch_size]
        batch_images = []
        batch_meta = []

        for pth in batch_files:
            fp = os.path.join(args.input_root, pth)
            img_bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
            if img_bgr is None:
                empty_mask = np.zeros((512, 512), dtype=np.uint8)
                cv2.imwrite(os.path.join(args.output_root, pth), empty_mask)
                cv2.imwrite(os.path.join(output_root_no_hair, pth), empty_mask)
                continue

            img_hwc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_hwc.shape[:2]
            batch_images.append(img_hwc)
            batch_meta.append((pth, h, w))

        if len(batch_images) == 0:
            continue

        with torch.inference_mode():
            all_faces = face_detector(
                batch_images,
                threshold=float(args.det_threshold),
                max_size=int(args.det_max_size),
                batch_size=len(batch_images),
            )
            all_faces = face_parser(batch_images, all_faces)

        for (pth, h, w), faces in zip(batch_meta, all_faces):
            if len(faces) == 0:
                mask_with_hair_np = np.zeros((h, w), dtype=np.float32)
                mask_no_hair_np = np.zeros((h, w), dtype=np.float32)
            else:
                areas = [
                    (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1])
                    for f in faces
                ]
                face = faces[int(np.argmax(areas))]
                seg_logits = face.get("seg_logits")
                if seg_logits is None:
                    mask_with_hair_np = np.zeros((h, w), dtype=np.float32)
                    mask_no_hair_np = np.zeros((h, w), dtype=np.float32)
                else:
                    seg_logits_t = torch.from_numpy(seg_logits).float()
                    seg_probs = F.softmax(seg_logits_t, dim=0)
                    face_prob_with_hair = seg_probs[1:11, :, :].sum(dim=0)
                    face_prob_no_hair = seg_probs[1:10, :, :].sum(dim=0)
                    mask_with_hair_np = face_prob_with_hair.numpy().astype(np.float32)
                    mask_no_hair_np = face_prob_no_hair.numpy().astype(np.float32)
                mask_with_hair_np = cv2.resize(mask_with_hair_np, (w, h), interpolation=cv2.INTER_LINEAR)
                mask_no_hair_np = cv2.resize(mask_no_hair_np, (w, h), interpolation=cv2.INTER_LINEAR)

            out_with_hair_mask = np.clip(mask_with_hair_np * 255.0, 0, 255).astype(np.uint8)
            out_no_hair_mask = np.clip(mask_no_hair_np * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(args.output_root, pth), out_with_hair_mask)
            cv2.imwrite(os.path.join(output_root_no_hair, pth), out_no_hair_mask)


if __name__ == "__main__":
    main()
