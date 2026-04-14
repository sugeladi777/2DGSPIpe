"""
Export mesh with mask-partitioned preprocessing + UV unwrapping using Blender.
Usage:
    blender --background --python export_uv_blender.py -- input.obj output.obj data_root
"""

from array import array
import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix


# Must match the face weighting atlas layout in texture/build_texture.py.
UV_PACK_MARGIN = 0.004
FACE_ATLAS_U_SPLIT = 0.30
REMOVE_DOUBLES_RATIO = 1e-6
DEGENERATE_FACE_AREA_RATIO = 1e-12
KEEP_ONLY_LARGEST_COMPONENT = True
FACE_MASK_SCORE_THRESHOLD = 0.35
FACE_OVER_HAIR_MARGIN = 0.05
MIN_VALID_MASK_SAMPLES = 1
MIN_PARTITION_COMPONENT_FACES = 128
MIN_PARTITION_COMPONENT_FACE_RATIO = 1e-3
PARTITION_CLEANUP_PASSES = 3
REGION_FACE = 1
REGION_HAIR = 2
HAIR_UV_RECT = (
    UV_PACK_MARGIN,
    UV_PACK_MARGIN,
    FACE_ATLAS_U_SPLIT - UV_PACK_MARGIN,
    1.0 - UV_PACK_MARGIN,
)
FACE_UV_RECT = (
    FACE_ATLAS_U_SPLIT + UV_PACK_MARGIN,
    UV_PACK_MARGIN,
    1.0 - UV_PACK_MARGIN,
    1.0 - UV_PACK_MARGIN,
)


def parse_script_args(argv: list[str]) -> tuple[str, str, str]:
    if "--" in argv:
        args = argv[argv.index("--") + 1 :]
    else:
        args = argv[-3:]
    if len(args) != 3:
        raise ValueError(
            "expected three args: input.obj output.obj data_root; "
            "usage: blender --background --python export_uv_blender.py -- input.obj output.obj data_root"
        )
    return args[0], args[1], args[2]


def _is_image_file(name: str) -> bool:
    return name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"))


def _find_mask_path(mask_root: str, frame_file_path: str) -> str:
    base = os.path.basename(frame_file_path)
    exact = os.path.join(mask_root, base)
    if os.path.isfile(exact):
        return exact

    stem = os.path.splitext(base)[0]
    if not stem:
        raise FileNotFoundError(f"cannot resolve mask for frame: {frame_file_path}")

    for name in os.listdir(mask_root):
        if os.path.splitext(name)[0] == stem and _is_image_file(name):
            return os.path.join(mask_root, name)
    raise FileNotFoundError(f"mask not found in {mask_root} for frame: {frame_file_path}")


def _load_uv_partition_inputs(data_root: str) -> tuple[dict, list[dict], str, str]:
    meta_path = os.path.join(data_root, "mesh", "transforms.json")
    face_mask_root = os.path.join(data_root, "face_mask")
    face_mask_no_hair_root = os.path.join(data_root, "face_mask_no_hair")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"camera transforms not found: {meta_path}")
    if not os.path.isdir(face_mask_root):
        raise FileNotFoundError(f"face_mask directory not found: {face_mask_root}")
    if not os.path.isdir(face_mask_no_hair_root):
        raise FileNotFoundError(f"face_mask_no_hair directory not found: {face_mask_no_hair_root}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    frames = meta.get("frames", [])
    if not frames:
        raise RuntimeError(f"no frames found in transforms: {meta_path}")
    for key in ("fl_x", "fl_y", "cx", "cy", "w", "h"):
        if key not in meta:
            raise KeyError(f"missing camera intrinsic key '{key}' in {meta_path}")

    return meta, frames, face_mask_root, face_mask_no_hair_root


class MaskImage:
    def __init__(self, path: str):
        image = bpy.data.images.load(path, check_existing=False)
        try:
            image.colorspace_settings.name = "Non-Color"
        except Exception:
            pass
        self.width = int(image.size[0])
        self.height = int(image.size[1])
        if self.width <= 0 or self.height <= 0:
            bpy.data.images.remove(image)
            raise RuntimeError(f"invalid mask image size: {path}")

        self.pixels = array("f", [0.0]) * (self.width * self.height * 4)
        image.pixels.foreach_get(self.pixels)
        bpy.data.images.remove(image)

    def sample(self, u: float, v: float) -> float | None:
        if u < 0.0 or v < 0.0 or u > self.width - 1 or v > self.height - 1:
            return None

        x0 = int(math.floor(u))
        y0_top = int(math.floor(v))
        x1 = min(x0 + 1, self.width - 1)
        y1_top = min(y0_top + 1, self.height - 1)
        tx = u - x0
        ty = v - y0_top

        # Blender stores image pixels bottom-up, while camera projection uses image top-left.
        y0 = self.height - 1 - y0_top
        y1 = self.height - 1 - y1_top

        v00 = self.pixels[(y0 * self.width + x0) * 4]
        v10 = self.pixels[(y0 * self.width + x1) * 4]
        v01 = self.pixels[(y1 * self.width + x0) * 4]
        v11 = self.pixels[(y1 * self.width + x1) * 4]
        return (
            (1.0 - tx) * (1.0 - ty) * v00
            + tx * (1.0 - ty) * v10
            + (1.0 - tx) * ty * v01
            + tx * ty * v11
        )


def _reset_scene() -> None:
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _import_obj(filepath: str) -> None:
    if not hasattr(bpy.ops.wm, "obj_import"):
        raise RuntimeError("This pipeline requires Blender 5.x (wm.obj_import not found).")
    bpy.ops.wm.obj_import(
        filepath=filepath,
        forward_axis="NEGATIVE_Z",
        up_axis="Y",
        import_vertex_groups=False,
        validate_meshes=True,
    )


def _export_obj(filepath: str) -> None:
    if not hasattr(bpy.ops.wm, "obj_export"):
        raise RuntimeError("This pipeline requires Blender 5.x (wm.obj_export not found).")
    bpy.ops.wm.obj_export(
        filepath=filepath,
        forward_axis="NEGATIVE_Z",
        up_axis="Y",
        export_selected_objects=True,
        export_uv=True,
        export_normals=True,
        export_materials=False,
    )


def _pick_single_mesh() -> bpy.types.Object:
    meshes = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not meshes:
        meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if len(meshes) != 1:
        raise RuntimeError(f"expected exactly one mesh object, got {len(meshes)}")
    bpy.ops.object.select_all(action="DESELECT")
    obj = meshes[0]
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    return obj


def _mesh_diag(bm: bmesh.types.BMesh) -> float:
    if not bm.verts:
        return 0.0
    xs = [float(v.co.x) for v in bm.verts]
    ys = [float(v.co.y) for v in bm.verts]
    zs = [float(v.co.z) for v in bm.verts]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    dz = max(zs) - min(zs)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _face_components(bm: bmesh.types.BMesh) -> list[list[bmesh.types.BMFace]]:
    components: list[list[bmesh.types.BMFace]] = []
    visited: set[int] = set()
    for face in bm.faces:
        if face.index in visited:
            continue
        stack = [face]
        visited.add(face.index)
        comp: list[bmesh.types.BMFace] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for edge in cur.edges:
                for linked in edge.link_faces:
                    if linked.index in visited:
                        continue
                    visited.add(linked.index)
                    stack.append(linked)
        components.append(comp)
    return components


def _preprocess_mesh(bm: bmesh.types.BMesh) -> tuple[int, int, int]:
    diag = _mesh_diag(bm)
    removed_degenerate = 0
    removed_components = 0
    removed_component_faces = 0

    merge_dist = max(0.0, diag * REMOVE_DOUBLES_RATIO)
    if merge_dist > 0.0:
        bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=merge_dist)

    area_eps = max(0.0, (diag * diag) * DEGENERATE_FACE_AREA_RATIO)
    deg_faces = [f for f in bm.faces if float(f.calc_area()) <= area_eps]
    if deg_faces:
        removed_degenerate = len(deg_faces)
        bmesh.ops.delete(bm, geom=deg_faces, context="FACES")

    if KEEP_ONLY_LARGEST_COMPONENT:
        comps = _face_components(bm)
        if comps:
            largest = max(comps, key=len)
            to_delete: list[bmesh.types.BMFace] = []
            for comp in comps:
                if comp is largest:
                    continue
                to_delete.extend(comp)
                removed_components += 1
                removed_component_faces += len(comp)
            if to_delete:
                bmesh.ops.delete(bm, geom=to_delete, context="FACES")

    loose_verts = [v for v in bm.verts if len(v.link_faces) == 0]
    if loose_verts:
        bmesh.ops.delete(bm, geom=loose_verts, context="VERTS")

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.faces.index_update()
    bm.edges.index_update()
    bm.verts.index_update()
    return removed_degenerate, removed_components, removed_component_faces


def _collect_face_samples(bm: bmesh.types.BMesh) -> list[tuple[int, tuple[float, float, float], tuple[float, float, float]]]:
    bm.normal_update()
    bm.faces.ensure_lookup_table()
    bm.faces.index_update()
    samples = []
    for face in bm.faces:
        center = face.calc_center_median()
        normal = face.normal
        samples.append(
            (
                int(face.index),
                (float(center.x), float(center.y), float(center.z)),
                (float(normal.x), float(normal.y), float(normal.z)),
            )
        )
    return samples


def _count_partition_labels(labels: dict[int, int]) -> dict[str, int]:
    return {
        "face_faces": sum(1 for label in labels.values() if label == REGION_FACE),
        "hair_faces": sum(1 for label in labels.values() if label == REGION_HAIR),
    }


def _build_face_neighbors(bm: bmesh.types.BMesh) -> dict[int, set[int]]:
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.index_update()
    neighbors = {int(face.index): set() for face in bm.faces}
    for edge in bm.edges:
        linked = [int(face.index) for face in edge.link_faces]
        for idx, face_index in enumerate(linked):
            for other_index in linked[idx + 1 :]:
                neighbors[face_index].add(other_index)
                neighbors[other_index].add(face_index)
    return neighbors


def _same_label_components(
    labels: dict[int, int],
    neighbors: dict[int, set[int]],
) -> list[tuple[int, list[int]]]:
    components: list[tuple[int, list[int]]] = []
    visited: set[int] = set()
    for start, label in labels.items():
        if start in visited:
            continue
        visited.add(start)
        stack = [start]
        component: list[int] = []
        while stack:
            cur = stack.pop()
            component.append(cur)
            for linked in neighbors.get(cur, ()):
                if linked in visited or labels.get(linked, REGION_HAIR) != label:
                    continue
                visited.add(linked)
                stack.append(linked)
        components.append((label, component))
    return components


def _dominant_border_label(
    component: list[int],
    current_label: int,
    labels: dict[int, int],
    neighbors: dict[int, set[int]],
    component_size_by_face: dict[int, int],
) -> int | None:
    component_set = set(component)
    border_counts: dict[int, int] = {}
    for face_index in component:
        for linked in neighbors.get(face_index, ()):
            if linked in component_set:
                continue
            label = labels.get(linked, REGION_HAIR)
            if label == current_label:
                continue
            if component_size_by_face.get(linked, 0) <= len(component):
                continue
            border_counts[label] = border_counts.get(label, 0) + 1
    if not border_counts:
        return None
    return max(border_counts.items(), key=lambda item: item[1])[0]


def _cleanup_partition_labels(
    bm: bmesh.types.BMesh,
    labels: dict[int, int],
) -> tuple[dict[int, int], dict[str, int]]:
    neighbors = _build_face_neighbors(bm)
    min_component_faces = max(
        MIN_PARTITION_COMPONENT_FACES,
        int(round(len(labels) * MIN_PARTITION_COMPONENT_FACE_RATIO)),
    )
    cleanup_stats = {
        "cleanup_min_component_faces": min_component_faces,
        "cleanup_passes": 0,
        "cleanup_components_relabelled": 0,
        "cleanup_faces_relabelled": 0,
    }

    cleaned = dict(labels)
    for pass_index in range(PARTITION_CLEANUP_PASSES):
        updates: dict[int, int] = {}
        components = _same_label_components(cleaned, neighbors)
        component_size_by_face = {
            face_index: len(component)
            for _, component in components
            for face_index in component
        }
        for label, component in components:
            if len(component) >= min_component_faces:
                continue
            target_label = _dominant_border_label(
                component,
                label,
                cleaned,
                neighbors,
                component_size_by_face,
            )
            if target_label is None or target_label == label:
                continue
            for face_index in component:
                updates[face_index] = target_label
            cleanup_stats["cleanup_components_relabelled"] += 1
            cleanup_stats["cleanup_faces_relabelled"] += len(component)

        if not updates:
            break
        cleaned.update(updates)
        cleanup_stats["cleanup_passes"] = pass_index + 1

    return cleaned, cleanup_stats


def _classify_faces_from_masks(
    bm: bmesh.types.BMesh,
    data_root: str,
) -> tuple[dict[int, int], dict[str, int]]:
    meta, frames, face_mask_root, face_mask_no_hair_root = _load_uv_partition_inputs(data_root)
    fx = float(meta["fl_x"])
    fy = float(meta["fl_y"])
    cx = float(meta["cx"])
    cy = float(meta["cy"])
    cam_width = float(meta["w"])
    cam_height = float(meta["h"])
    if cam_width <= 0.0 or cam_height <= 0.0:
        raise RuntimeError(f"invalid camera image size in transforms: w={cam_width}, h={cam_height}")

    face_samples = _collect_face_samples(bm)
    if not face_samples:
        raise RuntimeError("mesh has no faces after preprocessing")

    no_hair_score = {face_index: 0.0 for face_index, _, _ in face_samples}
    hair_score = {face_index: 0.0 for face_index, _, _ in face_samples}
    valid_count = {face_index: 0 for face_index, _, _ in face_samples}
    used_frames = 0

    for frame in frames:
        frame_file_path = frame.get("file_path", "")
        transform = frame.get("transform_matrix")
        if not frame_file_path or transform is None:
            continue

        with_hair_path = _find_mask_path(face_mask_root, frame_file_path)
        no_hair_path = _find_mask_path(face_mask_no_hair_root, frame_file_path)
        with_hair_mask = MaskImage(with_hair_path)
        no_hair_mask = MaskImage(no_hair_path)
        sx_with = with_hair_mask.width / cam_width
        sy_with = with_hair_mask.height / cam_height
        sx_no_hair = no_hair_mask.width / cam_width
        sy_no_hair = no_hair_mask.height / cam_height

        c2w = Matrix(transform)
        w2c = c2w.inverted()
        rows = tuple(tuple(float(w2c[i][j]) for j in range(4)) for i in range(3))
        cam_origin = (
            float(transform[0][3]),
            float(transform[1][3]),
            float(transform[2][3]),
        )

        for face_index, center, normal in face_samples:
            ox = cam_origin[0] - center[0]
            oy = cam_origin[1] - center[1]
            oz = cam_origin[2] - center[2]
            if ox * normal[0] + oy * normal[1] + oz * normal[2] <= 0.0:
                continue

            x = rows[0][0] * center[0] + rows[0][1] * center[1] + rows[0][2] * center[2] + rows[0][3]
            y = rows[1][0] * center[0] + rows[1][1] * center[1] + rows[1][2] * center[2] + rows[1][3]
            z = rows[2][0] * center[0] + rows[2][1] * center[1] + rows[2][2] * center[2] + rows[2][3]
            if z <= 1e-8:
                continue

            u = fx * x / z + cx
            v = fy * y / z + cy
            if u < 0.0 or v < 0.0 or u > cam_width - 1.0 or v > cam_height - 1.0:
                continue

            with_score = with_hair_mask.sample(u * sx_with, v * sy_with)
            no_hair = no_hair_mask.sample(u * sx_no_hair, v * sy_no_hair)
            if with_score is None or no_hair is None:
                continue

            no_hair = max(0.0, min(1.0, float(no_hair)))
            with_score = max(0.0, min(1.0, float(with_score)))
            no_hair_score[face_index] += no_hair
            hair_score[face_index] += max(0.0, with_score - no_hair)
            valid_count[face_index] += 1

        used_frames += 1

    if used_frames == 0:
        raise RuntimeError("no usable mask/camera frames found for UV partitioning")

    labels: dict[int, int] = {}
    stats = {"frames_used": used_frames, "unobserved_faces": 0}
    for face_index, _, _ in face_samples:
        count = valid_count[face_index]
        if count < MIN_VALID_MASK_SAMPLES:
            labels[face_index] = REGION_HAIR
            stats["unobserved_faces"] += 1
            continue

        face_mean = no_hair_score[face_index] / count
        hair_mean = hair_score[face_index] / count
        if face_mean >= FACE_MASK_SCORE_THRESHOLD and face_mean >= hair_mean + FACE_OVER_HAIR_MARGIN:
            label = REGION_FACE
        else:
            label = REGION_HAIR
        labels[face_index] = label

    raw_counts = _count_partition_labels(labels)
    if raw_counts["face_faces"] == 0:
        raise RuntimeError("mask partitioning produced no face faces")
    labels, cleanup_stats = _cleanup_partition_labels(bm, labels)
    clean_counts = _count_partition_labels(labels)
    if clean_counts["face_faces"] == 0:
        raise RuntimeError("mask partition cleanup removed all face faces")
    stats.update(
        {
            "raw_face_faces": raw_counts["face_faces"],
            "raw_hair_faces": raw_counts["hair_faces"],
            **clean_counts,
            **cleanup_stats,
        }
    )
    return labels, stats


def _mark_partition_seams(bm: bmesh.types.BMesh, labels: dict[int, int]) -> int:
    seam_count = 0
    for edge in bm.edges:
        edge.seam = False
        linked_labels = {labels.get(int(face.index), REGION_HAIR) for face in edge.link_faces}
        if len(linked_labels) > 1:
            edge.seam = True
            seam_count += 1
    return seam_count


def _select_region_faces(obj: bpy.types.Object, region: int, labels: dict[int, int]) -> int:
    bpy.ops.mesh.select_mode(type="FACE")
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.faces.index_update()
    for vert in bm.verts:
        vert.select_set(False)
    for edge in bm.edges:
        edge.select_set(False)
    selected = 0
    for face in bm.faces:
        should_select = labels.get(int(face.index), REGION_HAIR) == region
        face.select_set(should_select)
        if should_select:
            selected += 1
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    return selected


def _unwrap_selected_region(obj: bpy.types.Object, region: int, labels: dict[int, int]) -> int:
    selected = _select_region_faces(obj, region, labels)
    if selected == 0:
        return 0
    bpy.ops.uv.unwrap(method="ANGLE_BASED", fill_holes=True)
    bpy.ops.uv.minimize_stretch(iterations=30)
    return selected


def _uv_polygon_area(coords: list[tuple[float, float]]) -> float:
    if len(coords) < 3:
        return 0.0
    area = 0.0
    for idx, (u0, v0) in enumerate(coords):
        u1, v1 = coords[(idx + 1) % len(coords)]
        area += u0 * v1 - u1 * v0
    return abs(area) * 0.5


def _uv_edge_is_continuous(
    edge: bmesh.types.BMEdge,
    uv_layer: bmesh.types.BMLayerItem,
    eps: float = 1e-7,
) -> bool:
    if len(edge.link_faces) != 2:
        return False

    face_a, face_b = edge.link_faces
    uv_by_vert_a = {
        loop.vert.index: (float(loop[uv_layer].uv.x), float(loop[uv_layer].uv.y))
        for loop in face_a.loops
        if loop.vert in edge.verts
    }
    uv_by_vert_b = {
        loop.vert.index: (float(loop[uv_layer].uv.x), float(loop[uv_layer].uv.y))
        for loop in face_b.loops
        if loop.vert in edge.verts
    }
    if len(uv_by_vert_a) != 2 or len(uv_by_vert_b) != 2:
        return False

    for vert in edge.verts:
        vert_index = int(vert.index)
        uv_a = uv_by_vert_a.get(vert_index)
        uv_b = uv_by_vert_b.get(vert_index)
        if uv_a is None or uv_b is None:
            return False
        if abs(uv_a[0] - uv_b[0]) > eps or abs(uv_a[1] - uv_b[1]) > eps:
            return False
    return True


def _uv_islands_by_uv_continuity(
    bm: bmesh.types.BMesh,
    uv_layer: bmesh.types.BMLayerItem,
) -> list[list[bmesh.types.BMFace]]:
    bm.faces.ensure_lookup_table()
    bm.faces.index_update()
    islands: list[list[bmesh.types.BMFace]] = []
    visited: set[int] = set()
    for face in bm.faces:
        if int(face.index) in visited:
            continue
        visited.add(int(face.index))
        stack = [face]
        island: list[bmesh.types.BMFace] = []
        while stack:
            cur = stack.pop()
            island.append(cur)
            for edge in cur.edges:
                if not _uv_edge_is_continuous(edge, uv_layer):
                    continue
                for linked in edge.link_faces:
                    linked_index = int(linked.index)
                    if linked_index in visited:
                        continue
                    visited.add(linked_index)
                    stack.append(linked)
        islands.append(island)
    return islands


def _normalize_uv_island_texel_density(obj: bpy.types.Object) -> dict[str, float | int]:
    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.active
    if uv_layer is None:
        raise RuntimeError("UV layer not found after unwrap")

    islands = _uv_islands_by_uv_continuity(bm, uv_layer)
    metrics = []
    total_mesh_area = 0.0
    total_uv_area = 0.0
    for island in islands:
        mesh_area = sum(float(face.calc_area()) for face in island)
        uv_area = 0.0
        uv_sum_u = 0.0
        uv_sum_v = 0.0
        uv_count = 0
        for face in island:
            coords = []
            for loop in face.loops:
                uv = loop[uv_layer].uv
                coords.append((float(uv.x), float(uv.y)))
                uv_sum_u += float(uv.x)
                uv_sum_v += float(uv.y)
                uv_count += 1
            uv_area += _uv_polygon_area(coords)

        if mesh_area <= 0.0 or uv_area <= 1e-16 or uv_count == 0:
            continue
        metrics.append(
            {
                "faces": island,
                "mesh_area": mesh_area,
                "uv_area": uv_area,
                "center": (uv_sum_u / uv_count, uv_sum_v / uv_count),
            }
        )
        total_mesh_area += mesh_area
        total_uv_area += uv_area

    if total_mesh_area <= 0.0 or total_uv_area <= 0.0:
        return {"uv_density_islands": 0, "uv_density_scaled_islands": 0}

    scaled = 0
    for item in metrics:
        target_uv_area = total_uv_area * item["mesh_area"] / total_mesh_area
        scale = math.sqrt(max(target_uv_area, 1e-16) / item["uv_area"])
        if not math.isfinite(scale) or scale <= 0.0:
            continue
        center_u, center_v = item["center"]
        for face in item["faces"]:
            for loop in face.loops:
                uv = loop[uv_layer].uv
                uv.x = center_u + (uv.x - center_u) * scale
                uv.y = center_v + (uv.y - center_v) * scale
        scaled += 1

    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    return {
        "uv_density_islands": len(islands),
        "uv_density_scaled_islands": scaled,
        "uv_density_total_mesh_area": total_mesh_area,
        "uv_density_total_uv_area": total_uv_area,
    }


def _fit_region_uvs_to_rect(
    obj: bpy.types.Object,
    labels: dict[int, int],
    region: int,
    rect: tuple[float, float, float, float],
    stats_prefix: str,
) -> dict[str, float | int]:
    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.active
    if uv_layer is None:
        raise RuntimeError("UV layer not found before atlas fitting")

    uvs = [
        loop[uv_layer].uv
        for face in bm.faces
        if labels.get(int(face.index), REGION_HAIR) == region
        for loop in face.loops
    ]
    if not uvs:
        return {
            f"{stats_prefix}_fit_scale": 1.0,
            f"{stats_prefix}_fit_width": 0.0,
            f"{stats_prefix}_fit_height": 0.0,
            f"{stats_prefix}_fit_uvs": 0,
        }

    min_u = min(float(uv.x) for uv in uvs)
    max_u = max(float(uv.x) for uv in uvs)
    min_v = min(float(uv.y) for uv in uvs)
    max_v = max(float(uv.y) for uv in uvs)
    width = max(max_u - min_u, 1e-12)
    height = max(max_v - min_v, 1e-12)
    rect_u0, rect_v0, rect_u1, rect_v1 = rect
    target_width = max(rect_u1 - rect_u0, 1e-6)
    target_height = max(rect_v1 - rect_v0, 1e-6)
    scale = min(target_width / width, target_height / height)
    fitted_width = width * scale
    fitted_height = height * scale
    offset_u = rect_u0 + (target_width - fitted_width) * 0.5 - min_u * scale
    offset_v = rect_v0 + (target_height - fitted_height) * 0.5 - min_v * scale

    for uv in uvs:
        uv.x = float(uv.x) * scale + offset_u
        uv.y = float(uv.y) * scale + offset_v

    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    return {
        f"{stats_prefix}_fit_scale": scale,
        f"{stats_prefix}_fit_width": width,
        f"{stats_prefix}_fit_height": height,
        f"{stats_prefix}_fit_uvs": len(uvs),
    }


def _pack_region_to_rect(
    obj: bpy.types.Object,
    labels: dict[int, int],
    region: int,
    rect: tuple[float, float, float, float],
    stats_prefix: str,
) -> dict[str, float | int]:
    selected = _select_region_faces(obj, region, labels)
    if selected == 0:
        return {
            f"{stats_prefix}_packed_faces": 0,
            f"{stats_prefix}_fit_scale": 1.0,
            f"{stats_prefix}_fit_width": 0.0,
            f"{stats_prefix}_fit_height": 0.0,
            f"{stats_prefix}_fit_uvs": 0,
        }
    bpy.ops.uv.pack_islands(rotate=True, scale=False, margin=UV_PACK_MARGIN)
    stats = _fit_region_uvs_to_rect(obj, labels, region, rect, stats_prefix)
    stats[f"{stats_prefix}_packed_faces"] = selected
    return stats


def _unwrap_uv_by_partition(obj: bpy.types.Object, labels: dict[int, int]) -> dict[str, float | int]:
    region_counts = {
        "face_unwrapped": _unwrap_selected_region(obj, REGION_FACE, labels),
        "hair_unwrapped": _unwrap_selected_region(obj, REGION_HAIR, labels),
    }
    bpy.ops.mesh.select_all(action="SELECT")
    density_stats = _normalize_uv_island_texel_density(obj)
    face_fit_stats = _pack_region_to_rect(obj, labels, REGION_FACE, FACE_UV_RECT, "face")
    hair_fit_stats = _pack_region_to_rect(obj, labels, REGION_HAIR, HAIR_UV_RECT, "hair")
    bpy.ops.mesh.select_all(action="SELECT")
    region_counts.update(density_stats)
    region_counts.update(face_fit_stats)
    region_counts.update(hair_fit_stats)
    return region_counts


def export_uv(in_mesh_fpath: str, out_mesh_fpath: str, data_root: str) -> None:
    if not in_mesh_fpath.endswith(".obj"):
        raise ValueError(f"must use .obj format: {in_mesh_fpath}")
    if not out_mesh_fpath.endswith(".obj"):
        raise ValueError(f"must use .obj format: {out_mesh_fpath}")
    if not data_root or not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_root not found: {data_root}")

    _reset_scene()
    _import_obj(in_mesh_fpath)
    obj = _pick_single_mesh()

    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    removed_degenerate, removed_components, removed_component_faces = _preprocess_mesh(bm)
    labels, partition_stats = _classify_faces_from_masks(bm, os.path.abspath(data_root))
    partition_seams = _mark_partition_seams(bm, labels)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    unwrap_stats = _unwrap_uv_by_partition(obj, labels)

    print(
        "[UV] preprocess: removed_degenerate_faces=%d, removed_components=%d, removed_component_faces=%d"
        % (removed_degenerate, removed_components, removed_component_faces)
    )
    print(
        "[UV] mask_partition: frames_used=%d, face_faces=%d, hair_faces=%d, "
        "raw_face_faces=%d, raw_hair_faces=%d, unobserved_faces=%d, "
        "cleanup_min_component_faces=%d, cleanup_passes=%d, cleanup_components_relabelled=%d, "
        "cleanup_faces_relabelled=%d, partition_seams=%d"
        % (
            partition_stats["frames_used"],
            partition_stats["face_faces"],
            partition_stats["hair_faces"],
            partition_stats["raw_face_faces"],
            partition_stats["raw_hair_faces"],
            partition_stats["unobserved_faces"],
            partition_stats["cleanup_min_component_faces"],
            partition_stats["cleanup_passes"],
            partition_stats["cleanup_components_relabelled"],
            partition_stats["cleanup_faces_relabelled"],
            partition_seams,
        )
    )
    print(
        "[UV] unwrap: face=%d, hair=%d"
        % (
            unwrap_stats["face_unwrapped"],
            unwrap_stats["hair_unwrapped"],
        )
    )
    print(
        "[UV] texel_density: islands=%d, scaled_islands=%d, total_mesh_area=%.6g, total_uv_area=%.6g, "
        "face_fit_scale=%.6g, hair_fit_scale=%.6g, "
        "face_prefit_size=%.6gx%.6g, hair_prefit_size=%.6gx%.6g"
        % (
            unwrap_stats["uv_density_islands"],
            unwrap_stats["uv_density_scaled_islands"],
            unwrap_stats.get("uv_density_total_mesh_area", 0.0),
            unwrap_stats.get("uv_density_total_uv_area", 0.0),
            unwrap_stats.get("face_fit_scale", 1.0),
            unwrap_stats.get("hair_fit_scale", 1.0),
            unwrap_stats.get("face_fit_width", 0.0),
            unwrap_stats.get("face_fit_height", 0.0),
            unwrap_stats.get("hair_fit_width", 0.0),
            unwrap_stats.get("hair_fit_height", 0.0),
        )
    )
    print(
        "[UV] atlas_layout: hair_rect=(%.3f, %.3f, %.3f, %.3f), "
        "face_rect=(%.3f, %.3f, %.3f, %.3f)"
        % (*HAIR_UV_RECT, *FACE_UV_RECT)
    )

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    _export_obj(out_mesh_fpath)


if __name__ == "__main__":
    in_mesh, out_mesh, root = parse_script_args(sys.argv)
    export_uv(in_mesh, out_mesh, root)
