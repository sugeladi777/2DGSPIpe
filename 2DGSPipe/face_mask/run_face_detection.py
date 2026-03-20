import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm
import facer
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description="Generate soft face masks using pyfacer")
    parser.add_argument("--input_root", required=True, help="Path to input images")
    parser.add_argument("--output_root", required=True, help="Path to save face masks")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    face_detector = facer.face_detector('retinaface/mobilenet', device='cuda')
    face_parser = facer.face_parser('farl/lapa/448', device='cuda')

    input_files = sorted(os.listdir(args.input_root))
    for pth in tqdm(input_files):
        fp = os.path.join(args.input_root, pth)
        img_hwc = facer.read_hwc(fp)
        h, w = img_hwc.shape[:2]
        image_tensor = facer.hwc2bchw(img_hwc).to('cuda')

        with torch.no_grad():
            faces = face_detector(image_tensor)
            try:
                no_faces = (faces['rects'].shape[0] == 0)
            except Exception:
                no_faces = False

            if no_faces:
                mask_np = np.zeros((h, w), dtype=np.float32)
            else:
                faces = face_parser(image_tensor, faces)
                seg_logits = faces.get('seg', {}).get('logits', None)
                if seg_logits is None:
                    mask_np = np.zeros((h, w), dtype=np.float32)
                else:
                    seg_probs = F.softmax(seg_logits, dim=1)
                    face_prob = seg_probs[0, 1:10, :, :].sum(dim=0)
                    mask_np = face_prob.cpu().numpy().astype(np.float32)
                    mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)

        out_mask = np.clip(mask_np * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.output_root, pth), out_mask)


if __name__ == "__main__":
    main()