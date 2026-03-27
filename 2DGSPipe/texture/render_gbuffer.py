import os
import argparse
import numpy as np
import cv2
import trimesh
from tqdm import tqdm
import json
import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from mesh_renderer import MeshRenderer


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--device', type=str, default="0")
parser.add_argument('--syn', type=int, default=1)
parser.add_argument('--data_root', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--save_uv_vis', type=int, default=0)
parser.add_argument('--save_pho_mask', type=int, default=0)
opt = parser.parse_args()

opt.meta_file_path = os.path.join(opt.data_root, "transforms.json")

opt.device = "cuda:%s" % opt.device


def nerf_matrix_to_ngp(pose, scale=0.33):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ],dtype=pose.dtype)
    return new_pose


class MetaShapeDataset(Dataset):
    def __init__(self, meta_file_path, mode="train"):
        super().__init__()
        self.id_root = opt.data_root
        self.mode = mode

        with open(meta_file_path, 'r') as f:
            meta = json.load(f)

        self.HEIGHT = int(meta['h'])
        self.WIDTH = int(meta['w'])

        # load intrinsics
        self.intrinsic = np.eye(3, dtype=np.float32)
        self.intrinsic[0, 0] = meta['fl_x']
        self.intrinsic[1, 1] = meta['fl_y']
        self.intrinsic[0, 2] = meta['cx']
        self.intrinsic[1, 2] = meta['cy']
        self.intrinsic = torch.from_numpy(self.intrinsic)
        self.intrinsic_rel = torch.clone(self.intrinsic)
        self.intrinsic_rel[0] /= self.WIDTH
        self.intrinsic_rel[1] /= self.HEIGHT
        
        # split dataset
        frames = meta["frames"]
        frames = sorted(frames, key=lambda d: d['file_path'])
        if mode in ["val"]:
            frames = frames[::10]
        self.frames = frames
        self.num_frames = len(self.frames)

        # load per-frame meta information
        self.cam2world = []
        self.img_name_list = []
        for f_id in tqdm(range(self.num_frames)):
            cur_pose = np.array(frames[f_id]['transform_matrix'], dtype=np.float32)
            if opt.syn == 0:
                cur_pose = nerf_matrix_to_ngp(cur_pose, scale=1.)
            self.cam2world.append(cur_pose)
            img_name = os.path.basename(frames[f_id]['file_path'])
            self.img_name_list.append(img_name)
        
        self.cam2world = np.stack(self.cam2world, axis=0).astype(np.float32)  # [nf,4,4]
        self.cam2world = torch.from_numpy(self.cam2world)  # [nf,4,4]

        print("Create [%s] dataset, total [%d] frames" % (self.mode, self.num_frames))

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        c2w = self.cam2world[index]
        info = {
            "c2w": c2w,  # [4,4]
            "intrinsic": self.intrinsic,
            "intrinsic_rel": self.intrinsic_rel,
            "img_name": self.img_name_list[index],
        }
        return info


class DiffusionSampler:
    def __init__(self, opt):
        self.device = opt.device
        self.pin_memory = torch.cuda.is_available()
        self.mesh_renderer_pt3d = MeshRenderer(self.device)

        # set dataset
        train_data = MetaShapeDataset(
            meta_file_path=opt.meta_file_path, mode="train",
        )
        self.dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, pin_memory=self.pin_memory)
        self._load_geometry(os.path.join(opt.data_root, "final_hack.obj"))

        self.HEIGHT = train_data.HEIGHT
        self.WIDTH = train_data.WIDTH
        data_root = opt.data_root
        self.save_root = os.path.join(data_root, "uv")
        self.save_mask_root = os.path.join(data_root, "uv_mask")
        os.makedirs(self.save_root, exist_ok=True)
        os.makedirs(self.save_mask_root, exist_ok=True)
        self.save_vis_root = None
        self.save_pho_mask_root = None
        if opt.save_uv_vis:
            self.save_vis_root = os.path.join(data_root, "uv_vis")
            os.makedirs(self.save_vis_root, exist_ok=True)
        if opt.save_pho_mask:
            self.save_pho_mask_root = os.path.join(data_root, "pho_mask")
            os.makedirs(self.save_pho_mask_root, exist_ok=True)

    def _load_geometry(self, mesh_uv_path):
        device = self.device
        mesh = trimesh.load_mesh(mesh_uv_path)
        uv = torch.from_numpy(np.array(mesh.visual.uv, copy=True)).to(device).float()  # [v,2]
        vertices = torch.from_numpy(np.array(mesh.vertices, copy=True)).to(device).float()  # [v,3]
        self.vert_normal = torch.from_numpy(np.array(mesh.vertex_normals, copy=True)).to(device).float()  # [v,3]
        faces = torch.from_numpy(np.array(mesh.faces, copy=True)).to(device)  # [f,3]
        self.mesh = mesh
        self.uv = uv[None, ...]  # [1,v,2]
        self.uv = 2 * self.uv - 1
        self.uv[..., 1] *= -1
        self.vertices = vertices[None, ...]  # [1,v,3]
        self.faces = faces[None, ...]  # [1,v,3]

    def render(self):
        with torch.inference_mode():
            for data in tqdm(self.dataloader):
                c2w = data["c2w"].to(self.device, non_blocking=True)
                cam_int = data["intrinsic_rel"].to(self.device, non_blocking=True)
                img_names = data["img_name"]
                bs = c2w.shape[0]

                cam_ext = torch.inverse(c2w)[:, :3]  # [b,3,4]
                faces = self.faces.expand(bs, -1, -1)
                vertices = self.vertices.expand(bs, -1, -1)
                uv = self.uv.expand(bs, -1, -1)

                origins = c2w[:, :3, -1]
                viewdir = origins[:, None, :] - vertices
                mask = torch.sum(viewdir * self.vert_normal[None, ...], dim=-1, keepdim=True)
                mask = (mask > 0).float()

                attrs = torch.cat([uv, mask], dim=-1)
                mesh_dict = {
                    "faces": faces,
                    "vertice": vertices,
                    "attributes": attrs,
                    "size": (self.HEIGHT, self.WIDTH),
                }
                attr_img, _ = self.mesh_renderer_pt3d.render_mesh(mesh_dict, cam_int, cam_ext)
                uv_img = attr_img[:, :2]
                mask_img = attr_img[:, 2:3]

                uv_img_cpu = uv_img.cpu().half()
                mask_img_cpu = mask_img.cpu()
                uv_img_vis_cpu = None
                if self.save_vis_root is not None:
                    uv_img_vis = mask_img * (uv_img + 1) / 2
                    uv_img_vis = torch.cat([uv_img_vis, torch.zeros_like(mask_img)], dim=1)
                    uv_img_vis_cpu = uv_img_vis.cpu()

                for b_idx, raw_name in enumerate(img_names):
                    img_name = os.path.splitext(raw_name)[0]
                    mask_np = (mask_img_cpu[b_idx, 0].numpy() * 255.0).round().astype(np.uint8)
                    cv2.imwrite(os.path.join(self.save_mask_root, f"{img_name}.png"), mask_np)
                    torch.save(uv_img_cpu[b_idx:b_idx + 1], os.path.join(self.save_root, f"{img_name}.pkl"))
                    if uv_img_vis_cpu is not None:
                        save_image(uv_img_vis_cpu[b_idx], os.path.join(self.save_vis_root, f"{img_name}.jpg"))
                    if self.save_pho_mask_root is not None:
                        cv2.imwrite(os.path.join(self.save_pho_mask_root, f"{img_name}.png"), mask_np)


if __name__ == "__main__":
    ds = DiffusionSampler(opt)
    ds.render()
