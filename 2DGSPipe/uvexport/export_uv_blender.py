"""
Export mesh with UV unwrapping using Blender.
Usage:
    blender --background --python export_uv_blender.py -- input.obj output.obj
"""

import sys
import bpy


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

    # UV unwrap
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.unwrap(method="CONFORMAL")
    # Improve area distribution to reduce "large surface -> tiny UV island" issue
    bpy.ops.uv.average_islands_scale()
    bpy.ops.uv.pack_islands(rotate=True, margin=0.003)
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    _export_obj(out_mesh_fpath)


if __name__ == "__main__":
    in_mesh, out_mesh = parse_script_args(sys.argv)
    export_uv(in_mesh, out_mesh)
