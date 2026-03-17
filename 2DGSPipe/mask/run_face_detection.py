import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm
import facer
import torch


def main():
    """Generate face masks using pyfacer."""
    parser = argparse.ArgumentParser(
        description="Generate face masks for face-aware densification"
    )
    parser.add_argument("--input_root", required=True, help="Path to input images")
    parser.add_argument("--output_root", required=True, help="Path to save face masks")

    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    # Initialize face detector
    face_detector = facer.face_detector('retinaface/mobilenet', device='cuda')
    face_parser = facer.face_parser('farl/lapa/448', device='cuda')

    for pth in tqdm(sorted(os.listdir(args.input_root))):
        img_path = os.path.join(args.input_root, pth)
        image = cv2.imread(img_path)
        
        if image is None:
            # Save empty mask if image not found
            empty_mask = np.zeros((512, 512), dtype=np.uint8)
            cv2.imwrite(os.path.join(args.output_root, pth), empty_mask)
            continue

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = facer.hwc2bchw(facer.read_hwc(img_path)).to('cuda')

        # Detect faces
        with torch.no_grad():
            faces = face_detector(image_tensor)
            
            if faces['rects'].shape[0] == 0:
                # No face detected, save empty mask
                h, w = image.shape[:2]
                empty_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.imwrite(os.path.join(args.output_root, pth), empty_mask)
                continue
            
            # Parse face to get segmentation mask
            faces = face_parser(image_tensor, faces)
            
            # Get face segmentation mask (merge all face components)
            # labels: 0=background, 1=skin, 2=left_brow, 3=right_brow, 4=left_eye, 
            #         5=right_eye, 6=nose, 7=upper_lip, 8=inner_mouth, 9=lower_lip, 10=hair
            seg_logits = faces['seg']['logits']
            seg_labels = seg_logits.argmax(dim=1).cpu().numpy()[0]
            
            # Create binary mask (face region: skin + facial features + hair)
            face_mask = np.isin(seg_labels, [1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(np.uint8) * 255
            
            # Resize to original image size
            h, w = image.shape[:2]
            face_mask = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(os.path.join(args.output_root, pth), face_mask)


if __name__ == "__main__":
    import torch
    main()