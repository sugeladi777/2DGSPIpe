"""
Export mesh with UV unwrapping using Blender.
Usage:
    blender --background --python export_uv_blender.py -- input.obj output.obj
"""

import sys
import math
import bpy
import bmesh


# Keep CONFORMAL as the default unwrap method for downstream texture optimization.
# The main control knob is seam density: too many seams fragment long hair into
# many tiny islands and waste UV area during packing.
SEAM_ANGLE_CANDIDATES_DEG = (50.0, 65.0, 80.0)
MAX_SEAM_ISLANDS = 32
UV_PACK_MARGIN = 0.004


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


def parse_script_args(argv):
    if "--" in argv:
        script_args = argv[argv.index("--") + 1 :]
    else:
        script_args = argv[-2:]
    if len(script_args) != 2:
        raise ValueError(
            "expected two args: input.obj output.obj; "
            "usage: blender --background --python export_uv_blender.py -- input.obj output.obj"
        )
    return script_args[0], script_args[1]


def _clear_seams(bm: bmesh.types.BMesh) -> None:
    for edge in bm.edges:
        edge.seam = False


def _mark_structural_seams(bm: bmesh.types.BMesh, sharp_angle_deg: float) -> None:
    sharp_angle_rad = math.radians(sharp_angle_deg)
    for edge in bm.edges:
        # Keep topological boundaries split.
        if edge.is_boundary or not edge.is_manifold:
            edge.seam = True
            continue

        # Treat unsupported connectivity conservatively.
        if len(edge.link_faces) != 2:
            edge.seam = True
            continue

        try:
            face_angle = edge.calc_face_angle()
        except ValueError:
            edge.seam = True
            continue

        # Only cut genuinely sharp folds. Smooth hair/face regions stay connected.
        edge.seam = face_angle >= sharp_angle_rad


def _count_seam_islands(bm: bmesh.types.BMesh) -> int:
    visited = set()
    island_count = 0

    for face in bm.faces:
        if face.index in visited:
            continue
        island_count += 1
        stack = [face]
        visited.add(face.index)

        while stack:
            cur_face = stack.pop()
            for edge in cur_face.edges:
                if edge.seam:
                    continue
                for linked_face in edge.link_faces:
                    if linked_face.index in visited:
                        continue
                    visited.add(linked_face.index)
                    stack.append(linked_face)

    return island_count


def _choose_adaptive_seams(bm: bmesh.types.BMesh) -> tuple[float, int]:
    chosen_angle = SEAM_ANGLE_CANDIDATES_DEG[-1]
    island_count = 0

    for sharp_angle_deg in SEAM_ANGLE_CANDIDATES_DEG:
        _clear_seams(bm)
        _mark_structural_seams(bm, sharp_angle_deg)
        island_count = _count_seam_islands(bm)
        chosen_angle = sharp_angle_deg
        if island_count <= MAX_SEAM_ISLANDS:
            break

    return chosen_angle, island_count


def export_uv(in_mesh_fpath: str, out_mesh_fpath: str) -> None:
    if not in_mesh_fpath.endswith(".obj"):
        raise ValueError(f"must use .obj format: {in_mesh_fpath}")
    if not out_mesh_fpath.endswith(".obj"):
        raise ValueError(f"must use .obj format: {out_mesh_fpath}")

    # Clear scene robustly (do not assume default Camera/Cube/Light exist)
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    _import_obj(in_mesh_fpath)

    imported_meshes = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not imported_meshes:
        imported_meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if not imported_meshes:
        raise RuntimeError(f"no mesh object imported from {in_mesh_fpath}")
    if len(imported_meshes) != 1:
        raise RuntimeError(f"expected exactly one mesh object, got {len(imported_meshes)}")

    bpy.ops.object.select_all(action="DESELECT")
    obj = imported_meshes[0]
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # UV unwrap with constrained seam control.
    # We keep CONFORMAL because it preserves local structure better for later
    # texture optimization, but aggressively limit seam fragmentation.
    bpy.ops.object.mode_set(mode="EDIT")

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    chosen_angle_deg, island_count = _choose_adaptive_seams(bm)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    print(
        "[UV] unwrap=CONFORMAL, sharp_angle=%.1f deg, seam_islands=%d"
        % (chosen_angle_deg, island_count)
    )
    if island_count > MAX_SEAM_ISLANDS:
        print(
            "[UV] warning: seam island count (%d) is still high; packing may be less efficient."
            % island_count
        )

    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.unwrap(method="CONFORMAL", fill_holes=True)

    # Make texel density more uniform before packing.
    bpy.ops.uv.average_islands_scale()

    # Use a tight packing margin; a large margin wastes UV area on fragmented hair.
    bpy.ops.uv.pack_islands(rotate=True, margin=UV_PACK_MARGIN)

    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    _export_obj(out_mesh_fpath)


if __name__ == "__main__":
    in_mesh, out_mesh = parse_script_args(sys.argv)
    export_uv(in_mesh, out_mesh)
