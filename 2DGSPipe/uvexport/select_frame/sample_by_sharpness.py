import os
import argparse
import torch
import math
import json
import numpy as np


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img_root', type=str)
parser.add_argument('--cam_path', type=str)
parser.add_argument('--save_root', type=str)
parser.add_argument('--num_view', type=int, default=16)
parser.add_argument('--min_angle_diff', type=float, default=15.0, help='Minimum angle difference between selected frames')
opt, _ = parser.parse_known_args()


def compute_view_angle(transform_matrix):
    """
    Compute azimuth and elevation angles from camera transform matrix.
    
    Args:
        transform_matrix: 4x4 camera to world transformation matrix
    
    Returns:
        azimuth: angle in degrees [0, 360)
        elevation: angle in degrees [-90, 90]
    """
    # Extract camera position (translation vector)
    cam_pos = transform_matrix[:3, 3]
    
    # Compute spherical coordinates
    x, y, z = cam_pos[0], cam_pos[1], cam_pos[2]
    
    # Azimuth: angle in XY plane from X axis
    azimuth = math.atan2(y, x) * 180.0 / math.pi  # [-180, 180]
    if azimuth < 0:
        azimuth += 360.0  # [0, 360)
    
    # Elevation: angle from XY plane
    r = math.sqrt(x*x + y*y + z*z)
    elevation = math.asin(z / r) * 180.0 / math.pi  # [-90, 90]
    
    return azimuth, elevation


def angle_difference(angle1, angle2):
    """Compute minimum angle difference between two angles in degrees."""
    diff = abs(angle1 - angle2)
    if diff > 180:
        diff = 360 - diff
    return diff


# Load metadata
meta_file_path = opt.cam_path
with open(meta_file_path, 'r') as f:
    meta = json.load(f)

frames = meta["frames"]
frames = sorted(frames, key=lambda d: d['file_path'])
frame_names = set([os.path.basename(f['file_path']) for f in frames])

# Create directories
raw_frame_root = opt.img_root
img_root = os.path.join(opt.save_root, "image")
os.makedirs(img_root, exist_ok=True)

# Load sharpness information
sharp_info = torch.load(os.path.join(opt.save_root, "sharpness.pkl"))

# Filter valid frames
img_pth_list = sorted([f for f in os.listdir(raw_frame_root) if f in frame_names]) 
num_img = len(img_pth_list)

# Build frame info: compute view angles for each frame
frame_info = []
for i, img_name in enumerate(img_pth_list):
    # Find corresponding frame in meta
    frame_idx = None
    for j, f in enumerate(frames):
        if os.path.basename(f['file_path']) == img_name:
            frame_idx = j
            break
    
    if frame_idx is None:
        continue
    
    transform = np.array(frames[frame_idx]['transform_matrix'])
    azimuth, elevation = compute_view_angle(transform)
    sharpness = sharp_info.get(img_name, 0)
    
    frame_info.append({
        'img_name': img_name,
        'frame_idx': frame_idx,
        'azimuth': azimuth,
        'elevation': elevation,
        'sharpness': sharpness
    })

# Sort by azimuth to help with visualization
frame_info = sorted(frame_info, key=lambda x: x['azimuth'])

# Strategy: Select frames based on azimuth diversity + sharpness
# 1. Divide 360 degrees into num_view buckets
# 2. In each bucket, select the sharpest frame
# 3. Ensure minimum angle difference between selected frames

num_buckets = opt.num_view
bucket_size = 360.0 / num_buckets
min_angle_diff = opt.min_angle_diff

# Group frames into azimuth buckets
buckets = [[] for _ in range(num_buckets)]
for info in frame_info:
    bucket_idx = int(info['azimuth'] / bucket_size) % num_buckets
    buckets[bucket_idx].append(info)

# Select best frame from each bucket
selected_frames = []
selected_angles = []  # Track azimuth of selected frames for diversity check

for bucket_idx in range(num_buckets):
    if len(buckets[bucket_idx]) == 0:
        continue
    
    # Sort by sharpness (descending) and select the sharpest
    buckets[bucket_idx].sort(key=lambda x: x['sharpness'], reverse=True)
    best_frame = buckets[bucket_idx][0]
    
    # Check angle difference with already selected frames
    is_diverse = True
    for prev_angle in selected_angles:
        if angle_difference(best_frame['azimuth'], prev_angle) < min_angle_diff:
            is_diverse = False
            break
    
    # If not diverse enough, try next best frame in this bucket
    if not is_diverse:
        for fallback_frame in buckets[bucket_idx][1:]:
            is_diverse = True
            for prev_angle in selected_angles:
                if angle_difference(fallback_frame['azimuth'], prev_angle) < min_angle_diff:
                    is_diverse = False
                    break
            if is_diverse:
                best_frame = fallback_frame
                break
    
    selected_frames.append(best_frame)
    selected_angles.append(best_frame['azimuth'])

# Sort selected frames by sharpness (optional: process sharper frames first)
selected_frames.sort(key=lambda x: x['sharpness'], reverse=True)

# Save selected frames
new_frames = []
for info in selected_frames:
    img_name = info['img_name']
    src_path = os.path.join(raw_frame_root, img_name)
    dst_path = os.path.join(img_root, img_name)
    os.system("cp %s %s" % (src_path, dst_path))
    new_frames.append(frames[info['frame_idx']])

# Update metadata
meta["frames"] = new_frames
with open(os.path.join(opt.save_root, "select_sharp.json"), "w") as outfile:
    json.dump(meta, outfile, indent=4)

# Print summary
print(f"[Frame Selection] Selected {len(selected_frames)} frames")
print(f"[Frame Selection] Azimuth coverage: {sorted(selected_angles)}")
