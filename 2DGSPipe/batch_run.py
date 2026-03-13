import os
import subprocess
from multiprocessing import Pool

input_dir = "/home/lichengkai/RGB_Recon/input/mini_test"  # 视频文件夹路径
output_root = "/home/lichengkai/RGB_Recon/output/mini_test_more_frames"      # 保存结果的根目录
gpus = ["0", "1"]
base_port = 6009

# 获取所有视频文件
video_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.mp4')]
datasets = [os.path.join(output_root, os.path.splitext(os.path.basename(v))[0]) for v in video_paths]

def run_job(args):
    dataset, video, gpu, port = args
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["RECON_PORT"] = str(port)
    cmd = [
        "python", "/home/lichengkai/RGB_Recon/2DGSPipe/run.py",
        "--save_root", dataset,
        "--video_path", video,
    ]
    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    jobs = []
    for i, (dataset, video) in enumerate(zip(datasets, video_paths)):
        gpu = gpus[i % len(gpus)]
        port = base_port + i
        jobs.append((dataset, video, gpu, port))
    with Pool(processes=len(gpus)) as pool:
        pool.map(run_job, jobs)