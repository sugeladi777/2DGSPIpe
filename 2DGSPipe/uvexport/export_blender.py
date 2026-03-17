'''
Export mesh with UV unwrapping using Blender
Usage: blender --background --python export_blender.py input.obj output.obj
'''

import os
import sys
import bpy


def export_uv(in_mesh_fpath, out_mesh_fpath):
    assert in_mesh_fpath.endswith(".obj"), f"must use .obj format: {in_mesh_fpath}"
    assert out_mesh_fpath.endswith(".obj"), f"must use .obj format: {out_mesh_fpath}"

    bpy.data.objects["Camera"].select_set(True)
    bpy.data.objects["Cube"].select_set(True)
    bpy.data.objects["Light"].select_set(True)
    bpy.ops.object.delete()

    # 导入网格
    bpy.ops.import_scene.obj(
        filepath=in_mesh_fpath,
        use_edges=True,
        use_smooth_groups=True,
        use_split_objects=True,
        use_split_groups=True,
        use_groups_as_vgroups=False,
        use_image_search=True,
        split_mode="ON",
        global_clamp_size=0,
        axis_forward="-Z",
        axis_up="Y",
    )

    # 获取导入的对象（最后一个）
    obj = bpy.context.selected_objects[-1]
    bpy.context.view_layer.objects.active = obj
    
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    # bpy.ops.uv.unwrap()
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.export_scene.obj(
        filepath=out_mesh_fpath,
        axis_forward="-Z",
        axis_up="Y",
        use_selection=True,
        use_normals=True,
        use_uvs=True,
        use_materials=False,
        use_triangles=True,
    )
    

if __name__ == "__main__":
    # 使用位置参数
    in_mesh = sys.argv[-2]
    out_mesh = sys.argv[-1]
    
    export_uv(in_mesh, out_mesh)
