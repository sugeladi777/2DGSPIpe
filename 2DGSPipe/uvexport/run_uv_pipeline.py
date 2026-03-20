
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, required=True)
opt = parser.parse_args()

# 生成带UV坐标的网格
input_mesh = os.path.join(opt.data_root, "mesh", "2dgs_recon.obj")
uv_mesh = os.path.join(opt.data_root, "texture_dataset", "final_hack.obj")
os.makedirs(os.path.dirname(uv_mesh), exist_ok=True)

blender_bin = "/home/lichengkai/RGB_Recon/2DGSPipe/uvexport/blender-3.1.0-linux-x64/blender"
os.makedirs(os.path.dirname(uv_mesh), exist_ok=True)
os.system(f"{blender_bin} --background --python export_uv_blender.py {input_mesh} {uv_mesh}")

os.system(f"python select_frame/compute_sharpness.py --img_root {os.path.join(opt.data_root, 'raw_frames')} --save_root {os.path.join(opt.data_root, 'texture_dataset')}" )
os.system(f"python select_frame/sample_by_sharpness.py --img_root {os.path.join(opt.data_root, 'raw_frames')} --cam_path {os.path.join(opt.data_root, 'mesh', 'transforms.json')} --save_root {os.path.join(opt.data_root, 'texture_dataset')} --num_view 16")

# 准备纹理数据
os.system(f"python prepare_data.py --data_root {opt.data_root}" )
