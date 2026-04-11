import os
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--device', type=str, default="inherit")
parser.add_argument('--img_name', type=str, default="image")
parser.add_argument('--syn', type=int, default=1)
parser.add_argument('--vis_freq', type=int, default=50)
parser.add_argument('--data_root', type=str, required=True)
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--total_iter', type=int, default=151)
parser.add_argument('--weight_loss_tau', type=float, default=30.0)
parser.add_argument('--rgb_loss_weight', type=float, default=1.0)
parser.add_argument('--grad_loss_weight', type=float, default=5.0)
parser.add_argument('--lpips_loss_weight', type=float, default=0.05)
parser.add_argument('--lpips_start_iter', type=int, default=60)
parser.add_argument('--lpips_downsample', type=int, default=512)
parser.add_argument('--grad_use_view_weight', type=int, default=1)
parser.add_argument('--recompute_loss_weight', action='store_true')

opt = parser.parse_args()

opt.meta_file_path = os.path.join(opt.data_root, "transforms.json")

# Keep upstream GPU policy by default (e.g. 2DGSPipe/run.py --gpu auto).
# Only override CUDA_VISIBLE_DEVICES when explicitly requested.
_device_raw = str(opt.device).strip()
_device_key = _device_raw.lower()
if _device_key in {"", "inherit", "auto"}:
    pass
elif _device_key == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _device_raw


import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import lpips
from torch.utils.tensorboard import SummaryWriter
import kornia
import trimesh
import open3d as o3d

from module import InstantNGPNetwork
from mesh_renderer import MeshRenderer


net_cfg = {
    "log2_hashmap_size": 16,
    "finest_level": 2048,
    "uv_reso_w": 1536,
    "uv_reso_h": 1536,
    "batch_size": 2097152,
    "residual_scale": 0.25,
}


def compute_vertex_visibility_o3d(o3d_scene, point, uv_vertices):
    # 创建光线张量
    directions = uv_vertices - point
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    ray_origins = np.repeat(point[None, ...], len(uv_vertices), axis=0)
    rays = np.concatenate([ray_origins, directions], axis=1)
    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    
    # 计算交点个数
    intersection_counts = o3d_scene.count_intersections(rays)
    return intersection_counts.numpy()


class UVInstantNGP(nn.Module):
    '''
    Explicit UV texture map plus a small residual hash network.
    '''
    def __init__(self, cfg):
        super().__init__()
        self.width = cfg["uv_reso_w"]
        self.height = cfg["uv_reso_h"]
        self.cfg = cfg
        self.residual_scale = cfg.get("residual_scale", 0.25)

        # A learnable texture atlas acts as the main signal carrier.
        self.texture_map = nn.Parameter(torch.zeros(1, 3, self.height, self.width))

        finest_level = cfg["finest_level"]
        log2_hashmap_size = cfg["log2_hashmap_size"]
        self.residual_hashnet = InstantNGPNetwork(
            finest_level=finest_level, log2_hashmap_size=log2_hashmap_size, out_chns=3,
        )
        self.register_buffer("uv_coord", self._build_uv_coord(), persistent=False)

    def _build_uv_coord(self):
        x, y = torch.meshgrid(
            torch.arange(self.width, dtype=torch.float32),
            torch.arange(self.height, dtype=torch.float32),
            indexing="xy",
        )
        x = x.flatten() / self.width
        y = y.flatten() / self.height
        return torch.stack([x, y], dim=1)
    
    def _batch_forward(self, net):
        batch_size = self.cfg["batch_size"]
        res_list = []
        coord = self.uv_coord
        start = 0
        while start < coord.shape[0]:
            end = start + batch_size
            cur_coord = coord[start:end]
            cur_res = net(cur_coord)
            res_list.append(cur_res)
            start = end
        res_list = torch.cat(res_list, dim=0)
        res_list = res_list.reshape(self.height, self.width, -1)
        return res_list.permute(2, 0, 1)
        
    def get_base_texture(self):
        return torch.sigmoid(self.texture_map)

    def get_residual_texture(self):
        residual = self._batch_forward(self.residual_hashnet)
        residual = self.residual_scale * torch.tanh(residual)
        return residual[None, ...]

    def forward(self):
        base_texture = self.get_base_texture()
        residual_texture = self.get_residual_texture()
        return torch.clamp(base_texture + residual_texture, 0.0, 1.0)


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
    def __init__(self, device, meta_file_path):
        super().__init__()
        self.id_root = os.path.dirname(meta_file_path)
        self.device = device

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
        self.intrinsic = torch.from_numpy(self.intrinsic).to(self.device)
        self.intrinsic_rel = torch.clone(self.intrinsic)
        self.intrinsic_rel[0] /= self.WIDTH
        self.intrinsic_rel[1] /= self.HEIGHT
        
        # Texture optimization uses all available frames (no train/val split).
        all_frames = sorted(meta["frames"], key=lambda d: d['file_path'])
        frames = list(all_frames)
        self.frames = frames
        self.num_frames = len(self.frames)

        image_dir = os.path.join(self.id_root, opt.img_name)
        mask_dir = os.path.join(self.id_root, "uv_mask")
        uv_dir = os.path.join(self.id_root, "uv")

        # load per-frame meta information
        self.cam2world = []
        self.images, self.masks, self.uv_images = [], [], []
        self.img_idx = []
        self.img_name_list = []
        cnt = 0
        for f_id in tqdm(range(self.num_frames)):
            cur_pose = np.array(frames[f_id]['transform_matrix'], dtype=np.float32)
            if opt.syn == 0:
                cur_pose = nerf_matrix_to_ngp(cur_pose, scale=1.)
            self.cam2world.append(cur_pose)
            img_name = os.path.basename(frames[f_id]['file_path'])
            img_stem = os.path.splitext(img_name)[0]
            uv_img_name = "%s.pkl" % os.path.splitext(img_name)[0]

            img = self._load_img(os.path.join(image_dir, img_name))
            self.images.append(img)
            mask_path_png = os.path.join(mask_dir, f"{img_stem}.png")
            if not os.path.exists(mask_path_png):
                raise FileNotFoundError(
                    f"UV mask not found for frame '{img_name}': {mask_path_png}"
                )
            mask_path = mask_path_png
            mask = self._load_img(mask_path)
            self.masks.append(mask[:1])
            uv = torch.load(os.path.join(uv_dir, uv_img_name), map_location="cpu")[0].float()
            self.uv_images.append(uv)
            self.img_idx.append(cnt)
            self.img_name_list.append(img_name)
            cnt += 1

        self.images = torch.stack(self.images, dim=0)  # [b,3,h,w]
        self.masks = torch.stack(self.masks, dim=0)  # [b,1,h,w]
        self.uv_images = torch.stack(self.uv_images, dim=0)  # [b,2,h,w]
        
        self.cam2world = np.stack(self.cam2world, axis=0).astype(np.float32)  # [nf,4,4]
        self.cam2world = torch.from_numpy(self.cam2world).to(self.device)  # [nf,4,4]

        print("Create dataset, total [%d] frames" % (self.num_frames))

    def __len__(self):
        return self.num_frames
        
    def _load_img(self, pth):
        if os.path.exists(pth):
            img = transforms.ToTensor()(Image.open(pth))
            return img.to(self.device)
        else:
            return torch.zeros(3, self.HEIGHT, self.WIDTH).to(self.device)
        
    def __getitem__(self, index):
        img = self.images[index]
        render_mask = self.masks[index]
        if img.shape[0] == 4:
            render_mask = render_mask * img[3:]
            img = img[:3] * img[3:]
        
        uv_img = self.uv_images[index]

        c2w = self.cam2world[index]
        
        # origins, viewdirs = self._compute_ray(c2w)
        info = {
            "pixels": img,  # [3,h,w]
            "mask": render_mask,  # [1,h,w]
            "uv": uv_img,
            "c2w": c2w,  # [4,4]
            "intrinsic": self.intrinsic,
            "intrinsic_rel": self.intrinsic_rel,
            "idx": self.img_idx[index],
            "img_name": self.img_name_list[index],
            # "origins": origins[0],  # [3,h,w]
            # "viewdirs": viewdirs[0],  # [3,h,w]
        }
        return info


class DiffusionSampler:
    def __init__(self, opt):
        self.device = "cuda"
        self.log_dir = os.path.join(opt.data_root, "texture", opt.img_name)
        self.pin_memory = torch.cuda.is_available()

        # set dataset
        all_data = MetaShapeDataset(
            device="cpu", meta_file_path=opt.meta_file_path,
        )
        self.train_frame_count = all_data.num_frames
        self.batch_size = 4
        
        self.dataloader = DataLoader(all_data, batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_memory)
        self.weight_dataloader = DataLoader(all_data, batch_size=1, shuffle=False, pin_memory=self.pin_memory)
        self.val_dataloader = DataLoader(all_data, batch_size=1, shuffle=False, pin_memory=self.pin_memory)

        self.HEIGHT = all_data.HEIGHT  # image height
        self.WIDTH = all_data.WIDTH  # image width
        self.weight_loss_tau = opt.weight_loss_tau

        self.uv_reso_h = net_cfg["uv_reso_h"]
        self.uv_reso_w = net_cfg["uv_reso_w"]

        self.network = UVInstantNGP(net_cfg)
        self.network = self.network.to(self.device)

        self.lpips_loss = lpips.LPIPS(net="vgg").cuda()
        self.lpips_loss.requires_grad_(False)

        self.mesh_renderer = MeshRenderer(self.device)
        data_root = os.path.dirname(opt.meta_file_path)
        self.mesh_path = os.path.join(data_root, "final_hack.obj")
        self._load_geometry(self.mesh_path)
        self.loss_weight_cache_path = self._build_loss_weight_cache_path()

        self.loss_weight_img = self.load_or_compute_loss_weight()
        self.initialize_texture_map_from_views(all_data)
        print(
            "[Frames] using all frames for optimization: %d"
            % (self.train_frame_count)
        )

    def weighted_l1_loss(self, pred, target, weight):
        weight = weight.expand_as(pred)
        diff = torch.abs(pred - target) * weight
        denom = weight.sum().clamp_min(1e-6)
        return diff.sum() / denom

    def masked_l1_loss(self, pred, target, mask):
        mask = mask.expand_as(pred)
        diff = torch.abs(pred - target) * mask
        denom = mask.sum().clamp_min(1e-6)
        return diff.sum() / denom

    def downsample_for_lpips(self, pred, target):
        max_side = opt.lpips_downsample
        height, width = pred.shape[-2:]
        cur_max_side = max(height, width)
        if cur_max_side <= max_side:
            return pred, target

        scale = max_side / float(cur_max_side)
        out_h = max(1, int(round(height * scale)))
        out_w = max(1, int(round(width * scale)))
        pred_small = F.interpolate(pred, size=(out_h, out_w), mode="bilinear", align_corners=False)
        target_small = F.interpolate(target, size=(out_h, out_w), mode="bilinear", align_corners=False)
        return pred_small, target_small

    def build_texture_map_init(self, train_data):
        texel_count = self.uv_reso_h * self.uv_reso_w
        accum_color = torch.zeros(3, texel_count, dtype=torch.float32)
        accum_weight = torch.zeros(1, texel_count, dtype=torch.float32)
        global_color_sum = torch.zeros(3, dtype=torch.float32)
        global_weight_sum = torch.zeros(1, dtype=torch.float32)

        view_weight_list = self.loss_weight_img.detach().cpu()
        for img, mask, uv, view_weight in zip(
            train_data.images,
            train_data.masks,
            train_data.uv_images,
            view_weight_list,
        ):
            img = img.float()
            if img.shape[0] == 4:
                alpha = img[3:4]
                img = img[:3] * alpha
                mask = mask * alpha
            else:
                img = img[:3]

            pixel_weight = (mask.float() * view_weight.float()).reshape(-1)
            valid = pixel_weight > 1e-8
            if not torch.any(valid):
                continue

            colors = img.permute(1, 2, 0).reshape(-1, 3)[valid]
            uv_coord = uv.permute(1, 2, 0).reshape(-1, 2)[valid]
            pixel_weight = pixel_weight[valid]

            global_color_sum += torch.sum(colors * pixel_weight[:, None], dim=0)
            global_weight_sum += torch.sum(pixel_weight)

            u = (uv_coord[:, 0] + 1.0) * 0.5 * (self.uv_reso_w - 1)
            v = (uv_coord[:, 1] + 1.0) * 0.5 * (self.uv_reso_h - 1)

            x0 = torch.floor(u).long()
            y0 = torch.floor(v).long()
            x1 = torch.clamp(x0 + 1, max=self.uv_reso_w - 1)
            y1 = torch.clamp(y0 + 1, max=self.uv_reso_h - 1)

            wx1 = u - x0.float()
            wy1 = v - y0.float()
            wx0 = 1.0 - wx1
            wy0 = 1.0 - wy1

            for x_idx, y_idx, bilinear_weight in (
                (x0, y0, wx0 * wy0),
                (x0, y1, wx0 * wy1),
                (x1, y0, wx1 * wy0),
                (x1, y1, wx1 * wy1),
            ):
                splat_weight = pixel_weight * bilinear_weight
                flat_idx = y_idx * self.uv_reso_w + x_idx
                accum_weight.index_add_(1, flat_idx, splat_weight[None, :])
                accum_color.index_add_(1, flat_idx, colors.t() * splat_weight[None, :])

        if global_weight_sum.item() > 0:
            fill_color = global_color_sum / global_weight_sum.clamp_min(1e-6)
        else:
            fill_color = torch.full((3,), 0.5, dtype=torch.float32)

        init_texture = fill_color[:, None].repeat(1, texel_count)
        valid_texel = accum_weight[0] > 1e-8
        init_texture[:, valid_texel] = accum_color[:, valid_texel] / accum_weight[:, valid_texel]
        init_texture = init_texture.view(1, 3, self.uv_reso_h, self.uv_reso_w)

        covered_texel = int(valid_texel.sum().item())
        valid_uv_texel = int((self.uv_nonuse_mask[0, 0] < 0.5).sum().item())
        coverage_ratio = 0.0 if valid_uv_texel == 0 else covered_texel / float(valid_uv_texel)
        print(
            "[Texture Init] covered_uv_texels=%d/%d (%.2f%%)"
            % (covered_texel, valid_uv_texel, coverage_ratio * 100.0)
        )
        return init_texture

    def initialize_texture_map_from_views(self, train_data):
        with torch.no_grad():
            init_texture = self.build_texture_map_init(train_data)
            init_texture = init_texture.clamp(1e-4, 1.0 - 1e-4)
            init_param = torch.logit(init_texture)
            self.network.texture_map.copy_(init_param.to(self.device))

    def _write_textured_obj(self, src_obj_path, dst_obj_path, mtl_name, material_name="material_0"):
        with open(src_obj_path, "r") as f:
            lines = f.readlines()

        out_lines = []
        inserted = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("mtllib ") or stripped.startswith("usemtl "):
                continue

            if (not inserted) and (stripped.startswith("v ") or stripped.startswith("vt ") or stripped.startswith("vn ") or stripped.startswith("f ")):
                out_lines.append(f"mtllib {mtl_name}\n")
                out_lines.append(f"usemtl {material_name}\n")
                inserted = True
            out_lines.append(line)

        if not inserted:
            out_lines = [f"mtllib {mtl_name}\n", f"usemtl {material_name}\n"] + out_lines

        with open(dst_obj_path, "w") as f:
            f.writelines(out_lines)

    def export_textured_assets(self, uv_texture):
        data_root = os.path.dirname(opt.meta_file_path)
        texture_name = "final_hack_texture.png"
        mtl_name = "final_hack.mtl"
        material_name = "material_0"

        texture_path = os.path.join(data_root, texture_name)
        mtl_path = os.path.join(data_root, mtl_name)
        src_obj_path = os.path.join(data_root, "final_hack.obj")
        dst_obj_path = os.path.join(data_root, "final_textured.obj")

        # UV texture is [1, H, W, 3], save as image [1, 3, H, W]
        uv_img = uv_texture.permute(0, 3, 1, 2).detach().cpu().clamp(0.0, 1.0)
        save_image(uv_img, texture_path)

        with open(mtl_path, "w") as f:
            f.write(f"newmtl {material_name}\n")
            f.write("Ka 1.000000 1.000000 1.000000\n")
            f.write("Kd 1.000000 1.000000 1.000000\n")
            f.write("Ks 0.000000 0.000000 0.000000\n")
            f.write("d 1.000000\n")
            f.write("illum 2\n")
            f.write(f"map_Kd {texture_name}\n")

        self._write_textured_obj(src_obj_path, dst_obj_path, mtl_name, material_name)

        print(f"[Texture Export] texture: {texture_path}")
        print(f"[Texture Export] mtl: {mtl_path}")
        print(f"[Texture Export] textured obj: {dst_obj_path}")

    def _build_loss_weight_cache_path(self):
        tau_str = str(self.weight_loss_tau).replace(".", "p")
        return os.path.join(
            opt.data_root,
            f"loss_weight_tau{tau_str}_allviews{self.train_frame_count}_uv{self.uv_reso_h}x{self.uv_reso_w}.pt",
        )

    def _move_batch_to_device(self, data, keys=None):
        move_keys = data.keys() if keys is None else keys
        for k in move_keys:
            v = data[k]
            if torch.is_tensor(v):
                data[k] = v.to(self.device, non_blocking=self.pin_memory)
        return data

    def _loss_weight_dependencies(self):
        deps = [self.mesh_path, opt.meta_file_path]
        uv_root = os.path.join(opt.data_root, "uv")
        if os.path.isdir(uv_root):
            deps.extend(
                os.path.join(uv_root, name)
                for name in sorted(os.listdir(uv_root))
                if name.endswith(".pkl")
            )
        return deps

    def _is_loss_weight_cache_valid(self):
        if opt.recompute_loss_weight or not os.path.isfile(self.loss_weight_cache_path):
            return False
        cache_mtime = os.path.getmtime(self.loss_weight_cache_path)
        for dep in self._loss_weight_dependencies():
            if not os.path.isfile(dep):
                return False
            if os.path.getmtime(dep) > cache_mtime:
                return False
        return True

    def load_or_compute_loss_weight(self):
        if self._is_loss_weight_cache_valid():
            print(f"[Loss Weight] load cache: {self.loss_weight_cache_path}")
            cached = torch.load(self.loss_weight_cache_path, map_location="cpu")
            return cached.to(self.device, non_blocking=self.pin_memory)

        loss_weight_img = self.compute_loss_weight()
        torch.save(loss_weight_img.detach().cpu(), self.loss_weight_cache_path)
        print(f"[Loss Weight] save cache: {self.loss_weight_cache_path}")
        return loss_weight_img
    
    def _load_geometry(self, mesh_uv_path):
        device = self.device
        mesh = trimesh.load_mesh(mesh_uv_path)
        uv = torch.from_numpy(np.array(mesh.visual.uv, copy=True)).to(device).float()  # [v,2]
        vertices = torch.from_numpy(np.array(mesh.vertices, copy=True)).to(device).float()  # [v,3]
        normal = torch.from_numpy(np.array(mesh.vertex_normals, copy=True)).to(device).float()  # [v,3]
        normal = F.normalize(normal, dim=-1)
        faces = torch.from_numpy(np.array(mesh.faces, copy=True)).to(device)  # [f,3]
        self.mesh = mesh
        self.uv = uv[None, ...]  # [1,v,2]
        self.uv = 2 * self.uv - 1
        self.uv[..., 1] *= -1
        self.vertices = vertices[None, ...]  # [1,v,3]
        self.faces = faces[None, ...]  # [1,v,3]

        attrs = torch.cat([
            normal[None, ...], vertices[None, ...],
        ], dim=-1)

        # render the normal in UV space as the initialization
        uv_mesh_dict = {
            "faces": self.faces,
            "vertice": torch.cat([self.uv, torch.ones_like(self.uv[..., :1])], dim=-1),  # [1,v,3]
            "attributes": attrs,  # [1,v,3]
            "size": (self.uv_reso_h, self.uv_reso_w),
        }
        attrs_uv, uv_pix_to_face = self.mesh_renderer.render_ndc(uv_mesh_dict)  # [1,6,uv,uv]
        
        normal_uv = attrs_uv[:, :3]
        pos_uv = attrs_uv[:, 3:6]
        self.object_normal_lr_4k = torch.clone(normal_uv)  # [1,3,uh,uw]
        self.position_uv = torch.clone(pos_uv)  # [1,1,uh,uw]
        self.uv_nonuse_mask = (uv_pix_to_face == -1).float().permute(0, 3, 1, 2)  # [1,1,uh,uw]
        self.valid_uv_mask_flat = (self.uv_nonuse_mask[0, 0].reshape(-1) < 0.5)

    def render_in_uv(self, data, shading):
        data = self._move_batch_to_device(data, keys=("pixels", "mask", "uv", "idx"))
        uv_img = data["uv"]
        uv_img = uv_img.permute(0, 2, 3, 1).contiguous()
        shading_img = shading.permute(0, 3, 1, 2).expand(uv_img.shape[0], -1, -1, -1)
        
        tex_img = F.grid_sample(
            shading_img,
            uv_img, 
            align_corners=True
        )  # [b,3,h,w]   

        return tex_img

    def compute_loss_weight(self):
        uv_viewdir = []
        uv_list = []
        vis_list = []
        verts_uv = self.position_uv[0].permute(1, 2, 0).reshape(-1, 3)[self.valid_uv_mask_flat]
        verts_uv_np = verts_uv.detach().cpu().numpy()
        ks = 15
        kernel = torch.ones(ks, ks).to(self.device)

        mesh = o3d.io.read_triangle_mesh(self.mesh_path)
        mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_legacy)
        with torch.inference_mode():
            for data in tqdm(self.weight_dataloader):
                data = self._move_batch_to_device(data, keys=("c2w", "uv"))
                origins = data["c2w"][:, :3, -1]
                origins_np = origins[0].cpu().numpy()

                vis_mask = compute_vertex_visibility_o3d(
                    o3d_scene=scene, point=origins_np, uv_vertices=verts_uv_np,
                )
                vis_mask_flat = torch.zeros(self.uv_reso_h * self.uv_reso_w, device=self.device)
                vis_mask_flat[self.valid_uv_mask_flat] = torch.from_numpy(vis_mask).to(self.device).float()
                vis_mask_torch = vis_mask_flat.reshape(1, 1, self.uv_reso_h, self.uv_reso_w)
                vis_mask_torch = (vis_mask_torch <= 1).float()

                vis_mask_torch = kornia.morphology.erosion(vis_mask_torch, kernel=kernel)
                vis_list.append(vis_mask_torch)
                cur_uv_viewdir = F.normalize(self.position_uv - origins[..., None, None], dim=1)
                uv_viewdir.append(cur_uv_viewdir)
                uv_list.append(data["uv"])
        uv_viewdir = torch.cat(uv_viewdir, dim=0)
        uv_list = torch.cat(uv_list, dim=0)

        weight = torch.sum(-self.object_normal_lr_4k * uv_viewdir, dim=-3, keepdim=True)
        weight = torch.clamp(weight, min=0.)  # [nview,npatch,1,uh,uw]
        vis_list = torch.cat(vis_list, dim=0)
        weight = weight * vis_list
        # save_image(vis_list, "vis_uv.jpg")

        # softmax weight controlled by tau
        max_weight = F.softmax(self.weight_loss_tau * weight, dim=0)
        # save_image(max_weight, "weight.jpg")

        # for i in range(len(max_weight)):
        #     save_image(max_weight[i], os.path.join("workspace/weight_img_vis_erode", "%05d_weight.jpg" % i))

        uv_list = uv_list.permute(0, 2, 3, 1).contiguous()
        weight_img = F.grid_sample(max_weight, uv_list, align_corners=True)
        return weight_img

    def optimize(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=opt.lr)
        grad_scaler = torch.cuda.amp.GradScaler(2 ** 10)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=opt.total_iter,
            eta_min=opt.lr * 0.1,
        )
        writer = SummaryWriter(self.log_dir)
        print("[Optimizer] lr=%.6f" % opt.lr)
        
        for i in tqdm(range(opt.total_iter)):
            for data in self.dataloader:
                uv_shading = self.network().permute(0, 2, 3, 1).contiguous()
                render = self.render_in_uv(data, uv_shading)
                weight_img = self.loss_weight_img[data["idx"]]

                mask = data["mask"]
                gt = data["pixels"]

                img_pred = render * mask
                img_gt = gt * mask

                loss_rgb = self.masked_l1_loss(img_pred, img_gt, mask)
                grad_gt = kornia.filters.spatial_gradient(img_gt)
                grad_pred = kornia.filters.spatial_gradient(img_pred)
                if opt.grad_use_view_weight:
                    grad_weight = weight_img[:, :, None, ...] * mask[:, :, None, ...]
                    loss_grad = self.weighted_l1_loss(grad_pred, grad_gt, grad_weight)
                else:
                    loss_grad = self.masked_l1_loss(grad_pred, grad_gt, mask[:, :, None, ...])
                
                loss = opt.rgb_loss_weight * loss_rgb + opt.grad_loss_weight * loss_grad
                loss_lpips = torch.zeros((), device=self.device)
                if i >= opt.lpips_start_iter:
                    img_pred_lpips, img_gt_lpips = self.downsample_for_lpips(img_pred, img_gt)
                    loss_lpips = self.lpips_loss(img_pred_lpips, img_gt_lpips, normalize=True).mean()
                    loss = loss + opt.lpips_loss_weight * loss_lpips

                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                writer.add_scalar('loss/total', loss.item(), i)
                writer.add_scalar('loss/rgb', loss_rgb.item(), i)
                writer.add_scalar('loss/grad', loss_grad.item(), i)
                writer.add_scalar('loss/lpips_train', loss_lpips.item(), i)

            scheduler.step()
            writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], i)
            if i % opt.vis_freq == 0:
                cur_log_dir = os.path.join(self.log_dir, "%05d" % i)
                os.makedirs(cur_log_dir, exist_ok=True)
                with torch.no_grad():
                    uv_shading_eval = self.network().permute(0, 2, 3, 1).contiguous()
                    render_list = []
                    gt_list = []
                    psnr_list = []
                    ssim_list = []
                    lpips_list = []
                    for data in self.val_dataloader:
                        render = self.render_in_uv(data, uv_shading_eval)
                        mask = data["mask"]
                        gt = data["pixels"]
                        img_pred = render * mask
                        img_gt = gt * mask
                        gt_list.append(img_gt)
                        render_list.append(img_pred)
                        cur_psnr = kornia.metrics.psnr(img_pred, img_gt, max_val=1.).item()
                        cur_ssim = kornia.metrics.ssim(img_pred, img_gt, window_size=3).mean().item()
                        cur_lpips = self.lpips_loss(img_pred, img_gt, normalize=True).mean()
                        psnr_list.append(cur_psnr)
                        ssim_list.append(cur_ssim)
                        lpips_list.append(cur_lpips)
                    
                    render_list = torch.cat(render_list, dim=0)
                    gt_list = torch.cat(gt_list, dim=0)
                    save_image(render_list, os.path.join(cur_log_dir, "render.jpg"))
                    save_image(gt_list, os.path.join(cur_log_dir, "gt.jpg"))
                    save_image(uv_shading_eval.permute(0, 3, 1, 2), os.path.join(cur_log_dir, "uv_vis.png"))

                    psnr_score = sum(psnr_list) / len(psnr_list)
                    ssim_score = sum(ssim_list) / len(ssim_list)
                    lpips_score = sum(lpips_list) / len(lpips_list)
                    writer.add_scalar('ssim', ssim_score, i)
                    writer.add_scalar('psnr', psnr_score, i)
                    writer.add_scalar('lpips', lpips_score, i)
                    print("[iter %05d][PSNR %.4f][SSIM %.4f][LPIPS %.4f]" % (
                        i, psnr_score, ssim_score, lpips_score,
                    ))

        with torch.no_grad():
            final_uv_shading = self.network().permute(0, 2, 3, 1).contiguous()
            self.export_textured_assets(final_uv_shading)


if __name__ == "__main__":
    ir = DiffusionSampler(opt)
    ir.optimize()
