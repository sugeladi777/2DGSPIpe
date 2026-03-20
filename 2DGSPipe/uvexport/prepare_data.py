import os
import argparse
import json
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, required=True)
opt = parser.parse_args()

# 路径
save_root = os.path.join(opt.data_root, "texture_dataset")
# 创建目录并复制文件
os.makedirs(save_root, exist_ok=True)

# 过滤 transforms.json 中存在的帧
with open(os.path.join(opt.data_root, "mesh", "transforms.json"), 'r') as f:
    meta = json.load(f)

img_files = set(os.listdir(os.path.join(save_root,"image")))
meta["frames"] = [f for f in meta["frames"] if os.path.basename(f['file_path']) in img_files]

with open(os.path.join(save_root, "transforms.json"), "w") as f:
    json.dump(meta, f, indent=4)