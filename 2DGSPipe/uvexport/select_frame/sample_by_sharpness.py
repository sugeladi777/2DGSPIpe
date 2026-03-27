import argparse
import json
import math
import os
import shutil

import numpy as np
import torch
import trimesh


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--img_root", type=str, required=True)
parser.add_argument("--cam_path", type=str, required=True)
parser.add_argument("--save_root", type=str, required=True)
parser.add_argument("--mesh_path", type=str, default=None)
parser.add_argument("--num_view", type=int, default=16)
parser.add_argument(
    "--min_angle_diff",
    type=float,
    default=15.0,
    help="Minimum spherical angle difference in degrees between selected views",
)
opt = parser.parse_args()


def angle_difference(angle1, angle2):
    diff = abs(angle1 - angle2)
    if diff > 180.0:
        diff = 360.0 - diff
    return diff


def load_mesh_center(mesh_path):
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[0] == 0:
        raise RuntimeError(f"Failed to load mesh vertices from: {mesh_path}")
    bounds = np.stack([vertices.min(axis=0), vertices.max(axis=0)], axis=0)
    return bounds.mean(axis=0)


def compute_view_descriptor(transform_matrix, center):
    cam_pos = np.asarray(transform_matrix, dtype=np.float32)[:3, 3]
    direction = cam_pos - center
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        raise RuntimeError("Encountered camera position too close to the mesh center.")
    direction = direction / norm

    x, y, z = direction
    azimuth = math.degrees(math.atan2(y, x))
    if azimuth < 0.0:
        azimuth += 360.0
    elevation = math.degrees(math.asin(np.clip(z, -1.0, 1.0)))
    return cam_pos, direction.astype(np.float32), azimuth, elevation


def direction_angle_deg(direction_a, direction_b):
    dot = float(np.clip(np.dot(direction_a, direction_b), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def greedy_coverage_selection(frame_info, num_view, min_angle_diff):
    if num_view <= 0:
        raise ValueError("--num_view must be > 0")
    if not frame_info:
        raise RuntimeError("No frames available for coverage selection.")

    selected = [max(frame_info, key=lambda item: (item["sharpness"], -item["frame_idx"]))]
    selected_idx = {selected[0]["frame_idx"]}
    remaining = [item for item in frame_info if item["frame_idx"] not in selected_idx]

    while remaining and len(selected) < num_view:
        best_item = None
        best_key = None
        for item in remaining:
            min_dir_angle = min(direction_angle_deg(item["view_dir"], prev["view_dir"]) for prev in selected)
            min_azimuth = min(angle_difference(item["azimuth"], prev["azimuth"]) for prev in selected)
            min_elevation = min(abs(item["elevation"] - prev["elevation"]) for prev in selected)
            # Use spherical direction coverage as the hard selection target.
            # Azimuth/elevation are kept as tie-breakers so coverage is balanced in both dimensions.
            cur_key = (min_dir_angle, min_azimuth, min_elevation, item["sharpness"])
            if best_key is None or cur_key > best_key:
                best_key = cur_key
                best_item = item

        if best_item is None or best_key[0] < min_angle_diff:
            break

        selected.append(best_item)
        selected_idx.add(best_item["frame_idx"])
        remaining = [item for item in remaining if item["frame_idx"] not in selected_idx]

    return selected


meta_file_path = os.path.abspath(opt.cam_path)
if not os.path.isfile(meta_file_path):
    raise FileNotFoundError(f"Camera metadata not found: {meta_file_path}")

with open(meta_file_path, "r") as f:
    meta = json.load(f)

frames = sorted(meta["frames"], key=lambda item: item["file_path"])
frame_name_to_idx = {}
for idx, frame in enumerate(frames):
    frame_name = os.path.basename(frame["file_path"])
    if frame_name not in frame_name_to_idx:
        frame_name_to_idx[frame_name] = idx

raw_frame_root = os.path.abspath(opt.img_root)
if not os.path.isdir(raw_frame_root):
    raise FileNotFoundError(f"Image directory not found: {raw_frame_root}")

save_root = os.path.abspath(opt.save_root)
img_root = os.path.join(save_root, "image")
os.makedirs(img_root, exist_ok=True)

mesh_path = os.path.abspath(opt.mesh_path) if opt.mesh_path else os.path.join(save_root, "final_hack.obj")
if not os.path.isfile(mesh_path):
    raise FileNotFoundError(f"Mesh file not found for view selection: {mesh_path}")
mesh_center = load_mesh_center(mesh_path)

sharp_path = os.path.join(save_root, "sharpness.pkl")
if not os.path.isfile(sharp_path):
    raise FileNotFoundError(f"Sharpness file not found: {sharp_path}")
sharp_info = torch.load(sharp_path, map_location="cpu")

img_pth_list = sorted(
    [
        name
        for name in os.listdir(raw_frame_root)
        if os.path.isfile(os.path.join(raw_frame_root, name)) and name in frame_name_to_idx
    ]
)
if not img_pth_list:
    raise RuntimeError("No valid frames overlap between image directory and transforms.json")

frame_info = []
for img_name in img_pth_list:
    frame_idx = frame_name_to_idx.get(img_name)
    if frame_idx is None:
        continue

    transform = np.asarray(frames[frame_idx]["transform_matrix"], dtype=np.float32)
    cam_pos, view_dir, azimuth, elevation = compute_view_descriptor(transform, mesh_center)
    frame_info.append(
        {
            "img_name": img_name,
            "frame_idx": frame_idx,
            "cam_pos": cam_pos,
            "view_dir": view_dir,
            "azimuth": azimuth,
            "elevation": elevation,
            "sharpness": float(sharp_info.get(img_name, 0.0)),
        }
    )

selected_frames = greedy_coverage_selection(
    frame_info=frame_info,
    num_view=opt.num_view,
    min_angle_diff=opt.min_angle_diff,
)
selected_frames = sorted(selected_frames, key=lambda item: item["frame_idx"])
selected_img_names = {item["img_name"] for item in selected_frames}

for existing_name in os.listdir(img_root):
    existing_path = os.path.join(img_root, existing_name)
    if os.path.isfile(existing_path) and existing_name not in selected_img_names:
        os.remove(existing_path)

new_frames = []
for item in selected_frames:
    img_name = item["img_name"]
    src_path = os.path.join(raw_frame_root, img_name)
    dst_path = os.path.join(img_root, img_name)
    if not os.path.isfile(src_path):
        continue

    should_copy = True
    if os.path.isfile(dst_path):
        src_stat = os.stat(src_path)
        dst_stat = os.stat(dst_path)
        should_copy = (dst_stat.st_size != src_stat.st_size) or (dst_stat.st_mtime < src_stat.st_mtime)
    if should_copy:
        shutil.copy2(src_path, dst_path)
    new_frames.append(frames[item["frame_idx"]])

if not new_frames:
    raise RuntimeError("No frames were selected/copied. Please check input data and thresholds.")

meta["frames"] = new_frames
with open(os.path.join(save_root, "select_sharp.json"), "w") as outfile:
    json.dump(meta, outfile, indent=4)

selected_azimuth = [round(item["azimuth"], 3) for item in selected_frames]
selected_elevation = [round(item["elevation"], 3) for item in selected_frames]
if len(selected_frames) < opt.num_view:
    print(
        f"[Frame Selection] Hard coverage stopped at {len(selected_frames)} / {opt.num_view} "
        f"views because no remaining frame exceeded min_angle_diff={opt.min_angle_diff:.2f}"
    )
print(f"[Frame Selection] Mesh center: {mesh_center.tolist()}")
print(f"[Frame Selection] Selected {len(new_frames)} frames")
print(f"[Frame Selection] Azimuth coverage: {selected_azimuth}")
print(f"[Frame Selection] Elevation coverage: {selected_elevation}")
