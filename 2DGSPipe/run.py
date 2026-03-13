import os
import argparse
import datetime

# 命令行参数解析
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 视频相关参数
parser.add_argument('--video_path', type=str)  # 输入视频路径
parser.add_argument('--video_step_size', default=1, type=int)  # 帧提取间隔（每隔N帧取一帧）
parser.add_argument('--video_ds_ratio', default=1.0, type=float)  # 视频下采样比例

# 配准相关参数
parser.add_argument('--reg_close_eye', type=int, default=0)  # 是否使用闭眼模板（0=睁眼, 1=闭眼）

# 输出路径和选择执行的功能模块
parser.add_argument('--save_root', type=str)  # 结果保存根目录
parser.add_argument('--func', type=str, default="extract-mat-recon-refine-reg-tex")  # 执行的功能模块列表


opt, _ = parser.parse_known_args()
opt.save_root = os.path.abspath(opt.save_root)


# 创建输出目录
os.makedirs(opt.save_root, exist_ok=True)
# 生成日志文件名（带时间戳）
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(opt.save_root, f"log.txt")


# 写入日志（同时输出到控制台和文件）
def write_log(message):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")
    print(message)


# 记录开始时间
start_all = datetime.datetime.now()
write_log(f"Start Job: {opt.func}")
write_log(f"Start Time: {start_all.strftime('%Y-%m-%d %H:%M:%S')}")
write_log("-" * 30)


# 获取各模块代码路径
code_root = os.path.dirname(os.path.abspath(__file__))
mat_code_root = os.path.join(code_root, "matting")  # 前景分割模块
recon_code_root = os.path.join(code_root, "reconstruction")  # 重建模块
refine_code_root = os.path.join(code_root, "refinement")  # 精细化模块
reg_code_root = os.path.join(code_root, "registration")  # 配准模块
tex_code_root = os.path.join(code_root, "texture")  # 纹理模块

# 设置子目录路径
raw_frame_root = os.path.join(opt.save_root, "raw_frames")  # 原始帧保存路径
mask_save_root = os.path.join(opt.save_root, "mask")  # 掩码保存路径


# ========== 帧提取模块 ==========
if "extract" in opt.func:
    m_start = datetime.datetime.now()
    os.makedirs(raw_frame_root, exist_ok=True)
    # 使用ffmpeg提取视频帧，同时进行下采样
    extract_frame_cmd = """
        ffmpeg -i %s \\
            -vf "select=not(mod(n\\,%d)),scale=iw*%f:ih*%f,setsar=1:1" \\
            -vsync vfr -q:v 1 %s/%%05d.png
    """
    os.system(extract_frame_cmd % (opt.video_path, opt.video_step_size, opt.video_ds_ratio, opt.video_ds_ratio, raw_frame_root))
    
    m_end = datetime.datetime.now()
    write_log(f"[Module: extract] runtime: {m_end - m_start}")

# ========== 前景分割模块（Matting） ==========
if "mat" in opt.func:
    m_start = datetime.datetime.now()
    os.chdir(mat_code_root)
    os.system("python run_matting.py --input_root %s --output_root %s" % (raw_frame_root, mask_save_root))
    os.chdir(code_root)
    
    m_end = datetime.datetime.now()
    write_log(f"[Module: mat] runtime: {m_end - m_start}")

# ========== 重建模块（3D Gaussian Splatting） ==========
if "recon" in opt.func:
    m_start = datetime.datetime.now()
    os.chdir(recon_code_root)
    os.system("python run_reconstruction.py --data_root %s" % opt.save_root)
    os.chdir(code_root)
    
    m_end = datetime.datetime.now()
    write_log(f"[Module: recon] runtime: {m_end - m_start}")

# ========== 精细化模块（网格优化） ==========
if "refine" in opt.func:
    m_start = datetime.datetime.now()
    os.chdir(refine_code_root)
    os.system("python run_refinement.py --data_root %s" % opt.save_root)
    os.chdir(code_root)
    
    m_end = datetime.datetime.now()
    write_log(f"[Module: refine] runtime: {m_end - m_start}")

# ========== 配准模块（与模板对齐） ==========
if "reg" in opt.func:
    m_start = datetime.datetime.now()
    os.chdir(reg_code_root)
    os.system("python run_registration.py --data_root %s --close_eye %d" % (opt.save_root, opt.reg_close_eye))
    os.chdir(code_root)
    
    m_end = datetime.datetime.now()
    write_log(f"[Module: reg] runtime: {m_end - m_start}")

# ========== 纹理生成模块 ==========
if "tex" in opt.func:
    m_start = datetime.datetime.now()
    os.chdir(tex_code_root)
    os.system("python run_texture.py --data_root %s" % os.path.join(opt.save_root, "sample_dataset"))
    os.chdir(code_root)
    
    m_end = datetime.datetime.now()
    write_log(f"[Module: tex] runtime: {m_end - m_start}")

# 记录总运行时间
end_all = datetime.datetime.now()
write_log("-" * 30)
write_log(f"total runtime: {end_all - start_all}")
write_log(f"end time: {end_all.strftime('%Y-%m-%d %H:%M:%S')}")