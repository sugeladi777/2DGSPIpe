
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, required=True)
opt = parser.parse_args()

# 路径
input_mesh = os.path.join(opt.data_root, "refinement", "textured_mesh.obj")
uv_mesh = os.path.join(opt.data_root, "refinement", "textured_mesh_uv.obj")

# 1. 生成 UV
os.makedirs(os.path.dirname(uv_mesh), exist_ok=True)
os.system(f"blender --background --python export_blender.py {input_mesh} {uv_mesh}")

# 2. 准备纹理数据
os.system(f"python prepare_data.py --data_root {opt.data_root}")
