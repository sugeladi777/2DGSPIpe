"""
Export mesh with simple preprocessing + UV unwrapping using Blender.
Usage:
    blender --background --python export_uv_blender.py -- input.obj output.obj
"""

import math
import sys

import bmesh
import bpy


UV_PACK_MARGIN = 0.004
REMOVE_DOUBLES_RATIO = 1e-6
DEGENERATE_FACE_AREA_RATIO = 1e-12
KEEP_ONLY_LARGEST_COMPONENT = True


def parse_script_args(argv: list[str]) -> tuple[str, str]:
    if "--" in argv:
        args = argv[argv.index("--") + 1 :]
    else:
        args = argv[-2:]
    if len(args) != 2:
        raise ValueError(
            "expected two args: input.obj output.obj; "
            "usage: blender --background --python export_uv_blender.py -- input.obj output.obj"
        )
    return args[0], args[1]


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
    return removed_degenerate, removed_components, removed_component_faces


def _unwrap_uv() -> None:
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.unwrap(method="ANGLE_BASED", fill_holes=True)
    bpy.ops.uv.minimize_stretch(iterations=30)
    bpy.ops.uv.average_islands_scale()
    bpy.ops.uv.pack_islands(rotate=True, margin=UV_PACK_MARGIN)


def export_uv(in_mesh_fpath: str, out_mesh_fpath: str) -> None:
    if not in_mesh_fpath.endswith(".obj"):
        raise ValueError(f"must use .obj format: {in_mesh_fpath}")
    if not out_mesh_fpath.endswith(".obj"):
        raise ValueError(f"must use .obj format: {out_mesh_fpath}")

    _reset_scene()
    _import_obj(in_mesh_fpath)
    obj = _pick_single_mesh()

    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    removed_degenerate, removed_components, removed_component_faces = _preprocess_mesh(bm)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    _unwrap_uv()

    print(
        "[UV] preprocess: removed_degenerate_faces=%d, removed_components=%d, removed_component_faces=%d"
        % (removed_degenerate, removed_components, removed_component_faces)
    )

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    _export_obj(out_mesh_fpath)


if __name__ == "__main__":
    in_mesh, out_mesh = parse_script_args(sys.argv)
    export_uv(in_mesh, out_mesh)
