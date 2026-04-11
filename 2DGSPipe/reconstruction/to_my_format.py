import argparse
import json
import os
import glob
import re
import trimesh
import numpy as np


def _parse_iter_from_parent(mesh_path: str) -> int:
    parent = os.path.basename(os.path.dirname(mesh_path))
    matched = re.search(r"_(\d+)$", parent)
    if matched:
        return int(matched.group(1))
    return -1


def find_best_mesh_path(data_root: str) -> str:
    train_root = os.path.join(data_root, "recon", "train")
    candidates = glob.glob(os.path.join(train_root, "**", "fuse_post.ply"), recursive=True)
    if not candidates:
        candidates = glob.glob(os.path.join(data_root, "recon", "**", "fuse_post.ply"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"Cannot find fuse_post.ply under: {os.path.join(data_root, 'recon')}")

    candidates = sorted(
        candidates,
        key=lambda p: (_parse_iter_from_parent(p), os.path.getmtime(p)),
    )
    return candidates[-1]


def to_frame_path(data_root: str, img_name: str) -> str:
    raw_root = os.path.join(data_root, "raw_frames")
    stem_or_name = str(img_name)
    stem, ext = os.path.splitext(stem_or_name)

    if ext:
        exact = os.path.join(raw_root, stem_or_name)
        if os.path.isfile(exact):
            return exact
        stem_or_name = stem

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
    for candidate_ext in exts:
        candidate = os.path.join(raw_root, f"{stem_or_name}{candidate_ext}")
        if os.path.isfile(candidate):
            return candidate

    return os.path.join(raw_root, f"{stem_or_name}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    opt = parser.parse_args()

    data_root = os.path.abspath(opt.data_root)
    mesh_path = find_best_mesh_path(data_root)
    mesh = trimesh.load(mesh_path)

    mesh_out_root = os.path.join(data_root, "mesh")
    os.makedirs(mesh_out_root, exist_ok=True)
    mesh.export(os.path.join(mesh_out_root, "2dgs_recon.obj"))

    cam_path = os.path.join(data_root, "recon", "cameras.json")
    if not os.path.isfile(cam_path):
        raise FileNotFoundError(f"Camera file not found: {cam_path}")

    with open(cam_path, 'r') as f:
        cam_info_list = json.load(f)
    if not cam_info_list:
        raise RuntimeError(f"Empty camera list in: {cam_path}")

    save_info = {
        "fl_x": cam_info_list[0]["fx"],
        "fl_y": cam_info_list[0]["fy"],
        "w": cam_info_list[0]["width"],
        "h": cam_info_list[0]["height"],
    }
    save_info["cx"] = save_info["w"] / 2
    save_info["cy"] = save_info["h"] / 2
    frames = []

    for cam_info in cam_info_list:
        trans = np.asarray(cam_info["position"], dtype=np.float32)
        rot = np.asarray(cam_info["rotation"], dtype=np.float32)
        if trans.shape != (3,) or rot.shape != (3, 3):
            raise ValueError(f"Invalid camera pose shape for image: {cam_info.get('img_name', 'unknown')}")

        cur_frame = {
            "file_path": to_frame_path(data_root, cam_info["img_name"])
        }
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rot
        transform[:3, 3] = trans
        cur_frame["transform_matrix"] = transform.tolist()
        frames.append(cur_frame)

    save_info["frames"] = frames
    with open(os.path.join(mesh_out_root, "transforms.json"), 'w') as f:
        json.dump(save_info, f, indent=4)


if __name__ == "__main__":
    main()
