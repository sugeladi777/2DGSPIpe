import argparse
import csv
import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from batch_face import RetinaFace


MODEL_POINTS_5 = np.array(
    [
        (-30.0, 35.0, 30.0),   # left eye
        (30.0, 35.0, 30.0),    # right eye
        (0.0, 0.0, 0.0),       # nose tip
        (-25.0, -35.0, 20.0),  # left mouth corner
        (25.0, -35.0, 20.0),   # right mouth corner
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video_path", required=True, help="输入视频路径")
    parser.add_argument("--output_root", required=True, help="抽帧输出目录")
    parser.add_argument("--gpu_id", type=int, default=0, help="RetinaFace 使用的 GPU ID")
    parser.add_argument("--candidate_step_size", type=int, default=10, help="候选帧步长")
    parser.add_argument("--video_ds_ratio", type=float, default=0.5, help="解码后缩放比例")
    parser.add_argument("--min_face_size", type=float, default=48.0, help="最小人脸边长")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="全局选帧数量上限；当 target_num_frames<=0 时作为备用数量，<=0 表示不限制",
    )
    parser.add_argument(
        "--target_num_frames",
        type=int,
        default=70,
        help="全局选帧固定数量；>0 时优先于 max_frames",
    )
    return parser.parse_args()


def euler_from_rotation_matrix(rot_mat: np.ndarray) -> np.ndarray:
    sy = math.sqrt(rot_mat[0, 0] * rot_mat[0, 0] + rot_mat[1, 0] * rot_mat[1, 0])
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(rot_mat[2, 1], rot_mat[2, 2])
        yaw = math.atan2(-rot_mat[2, 0], sy)
        roll = math.atan2(rot_mat[1, 0], rot_mat[0, 0])
    else:
        pitch = math.atan2(-rot_mat[1, 2], rot_mat[1, 1])
        yaw = math.atan2(-rot_mat[2, 0], sy)
        roll = 0.0

    return np.degrees([yaw, pitch, roll]).astype(np.float32)


def estimate_head_pose(kps: np.ndarray, frame_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    height, width = frame_shape[:2]
    focal_length = float(max(width, height))
    center = (width * 0.5, height * 0.5)
    camera_matrix = np.array(
        [
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    image_points = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
    if image_points.shape != (5, 2):
        return None

    ok, rotation_vec, _ = cv2.solvePnP(
        MODEL_POINTS_5,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok:
        return None

    rot_mat, _ = cv2.Rodrigues(rotation_vec)
    return euler_from_rotation_matrix(rot_mat)


def select_largest_face(faces) -> Optional[Dict]:
    if not faces:
        return None
    areas = []
    for face in faces:
        box = np.asarray(face["box"], dtype=np.float32)
        areas.append(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))
    if not areas:
        return None
    return faces[int(np.argmax(areas))]


def maybe_resize(frame_bgr: np.ndarray, ds_ratio: float) -> np.ndarray:
    if ds_ratio >= 0.999:
        return frame_bgr
    return cv2.resize(frame_bgr, None, fx=ds_ratio, fy=ds_ratio, interpolation=cv2.INTER_AREA)


def compute_pose_delta(
    pose: Optional[np.ndarray],
    ref_pose: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], float]:
    if pose is None or ref_pose is None:
        return None, 0.0
    delta = np.abs(pose - ref_pose)
    delta = np.minimum(delta, 360.0 - delta)
    weighted = max(
        float(delta[0] / 10.0),
        float(delta[1] / 8.0),
        float(delta[2] / 10.0),
    )
    return delta, weighted


def pose_distance_score(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    delta = np.abs(pose_a - pose_b)
    delta = np.minimum(delta, 360.0 - delta)
    scaled = np.array([delta[0] / 10.0, delta[1] / 8.0, delta[2] / 10.0], dtype=np.float32)
    return float(np.linalg.norm(scaled))


def select_pose_coverage_candidates(records: List[Dict], target_num_frames: int) -> List[int]:
    if target_num_frames <= 0 or len(records) <= target_num_frames:
        return list(range(len(records)))

    valid_indices = [idx for idx, rec in enumerate(records) if rec.get("pose") is not None]
    if not valid_indices:
        return list(range(target_num_frames))
    if target_num_frames == 1:
        pose_stack = np.stack([np.asarray(records[idx]["pose"], dtype=np.float32) for idx in valid_indices], axis=0)
        pose_center = pose_stack.mean(axis=0)
        best_idx = max(
            valid_indices,
            key=lambda idx: pose_distance_score(np.asarray(records[idx]["pose"], dtype=np.float32), pose_center),
        )
        return [best_idx]
    if len(valid_indices) == 1:
        selected = [valid_indices[0]]
    else:
        selected = []
        max_pair_dist = -1.0
        for i, idx_a in enumerate(valid_indices[:-1]):
            pose_a = np.asarray(records[idx_a]["pose"], dtype=np.float32)
            for idx_b in valid_indices[i + 1 :]:
                pose_b = np.asarray(records[idx_b]["pose"], dtype=np.float32)
                pair_dist = pose_distance_score(pose_a, pose_b)
                if pair_dist > max_pair_dist:
                    max_pair_dist = pair_dist
                    selected = [idx_a, idx_b]
        if not selected:
            selected = [valid_indices[0]]

    selected_set = set(selected)
    while len(selected) < min(target_num_frames, len(valid_indices)):
        best_idx = None
        best_min_dist = -1.0
        for idx in valid_indices:
            if idx in selected_set:
                continue
            pose = np.asarray(records[idx]["pose"], dtype=np.float32)
            min_dist = min(
                pose_distance_score(pose, np.asarray(records[sel_idx]["pose"], dtype=np.float32))
                for sel_idx in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        selected_set.add(best_idx)

    if len(selected) < target_num_frames:
        remaining = [idx for idx in range(len(records)) if idx not in selected_set]
        remaining.sort(key=lambda idx: float(records[idx].get("pose_score", 0.0)), reverse=True)
        for idx in remaining:
            selected.append(idx)
            selected_set.add(idx)
            if len(selected) >= target_num_frames:
                break

    selected.sort(key=lambda idx: records[idx]["source_frame_idx"])
    return selected[:target_num_frames]


def resolve_selection_count(records: List[Dict], target_num_frames: int, max_frames: int) -> int:
    available = len(records)
    if target_num_frames > 0:
        return min(target_num_frames, available)
    if max_frames > 0:
        return min(max_frames, available)
    return available


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    csv_path = os.path.join(args.output_root, "selection_log.csv")

    detector = RetinaFace(gpu_id=args.gpu_id, network="resnet50", return_dict=True)
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"[adaptive_extract] start video={args.video_path}, total_frames={total_frames}, "
        f"candidate_step_size={args.candidate_step_size}, ds_ratio={args.video_ds_ratio:.4f}, "
        f"target_num_frames={args.target_num_frames}, max_frames={args.max_frames}, "
        "selection=pose_coverage",
        flush=True,
    )

    stats = {
        "candidates": 0,
        "detected": 0,
        "saved": 0,
        "saved_global": 0,
        "skipped_no_face": 0,
        "skipped_small_face": 0,
    }

    last_pose = None
    frame_idx = 0
    records: List[Dict] = []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "output_name",
                "source_frame_idx",
                "reason",
                "yaw",
                "pitch",
                "roll",
                "delta_yaw",
                "delta_pitch",
                "delta_roll",
                "pose_score",
            ],
        )
        writer.writeheader()

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_idx % args.candidate_step_size != 0:
                frame_idx += 1
                continue

            stats["candidates"] += 1
            frame_bgr = maybe_resize(frame_bgr, args.video_ds_ratio)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            faces_batch = detector([frame_rgb], threshold=0.6, max_size=max(frame_bgr.shape[:2]), batch_size=1)
            faces = faces_batch[0] if faces_batch else []
            face = select_largest_face(faces)

            if face is None:
                stats["skipped_no_face"] += 1
                frame_idx += 1
                continue

            stats["detected"] += 1
            box = np.asarray(face["box"], dtype=np.float32)
            face_w = float(box[2] - box[0])
            face_h = float(box[3] - box[1])
            if min(face_w, face_h) < args.min_face_size:
                stats["skipped_small_face"] += 1
                frame_idx += 1
                continue

            pose = estimate_head_pose(np.asarray(face["kps"], dtype=np.float32), frame_bgr.shape)
            pose_delta, pose_score = compute_pose_delta(pose, last_pose)

            record = {
                "frame_bgr": frame_bgr,
                "source_frame_idx": frame_idx,
                "pose": pose,
                "pose_delta": pose_delta,
                "pose_score": pose_score,
            }
            records.append(record)
            last_pose = pose

            frame_idx += 1

        selection_count = resolve_selection_count(records, args.target_num_frames, args.max_frames)
        print(
            f"[adaptive_extract] valid_candidates={len(records)}, selection_count={selection_count}",
            flush=True,
        )
        selected_indices = select_pose_coverage_candidates(records, selection_count)
        for save_idx, rec_idx in enumerate(selected_indices, start=1):
            rec = records[rec_idx]
            output_name = f"{save_idx:05d}.png"
            output_path = os.path.join(args.output_root, output_name)
            cv2.imwrite(output_path, rec["frame_bgr"])

            stats["saved"] += 1
            stats["saved_global"] += 1

            pose = rec["pose"]
            pose_delta = rec["pose_delta"]
            writer.writerow(
                {
                    "output_name": output_name,
                    "source_frame_idx": rec["source_frame_idx"],
                    "reason": "pose_coverage",
                    "yaw": "" if pose is None else f"{pose[0]:.4f}",
                    "pitch": "" if pose is None else f"{pose[1]:.4f}",
                    "roll": "" if pose is None else f"{pose[2]:.4f}",
                    "delta_yaw": "" if pose_delta is None else f"{pose_delta[0]:.4f}",
                    "delta_pitch": "" if pose_delta is None else f"{pose_delta[1]:.4f}",
                    "delta_roll": "" if pose_delta is None else f"{pose_delta[2]:.4f}",
                    "pose_score": f"{float(rec['pose_score']):.6f}",
                }
            )

    cap.release()
    print(
        (
            "[adaptive_extract] done "
            f"candidates={stats['candidates']}, detected={stats['detected']}, saved={stats['saved']}, "
            f"saved_global={stats['saved_global']}, "
            f"skipped_no_face={stats['skipped_no_face']}, skipped_small_face={stats['skipped_small_face']}"
        ),
        flush=True,
    )
    print(f"[adaptive_extract] selection_log={csv_path}", flush=True)


if __name__ == "__main__":
    main()
