# RGB Recon H5 + Backend 部署与访问（服务器2）

本文档按你当前服务器配置编写：
- `HostName`: `166.111.86.40`
- `SSH Port`: `8107`
- `User`: `lichengkai`
- 当前假设：仅 SSH 端口可入站，不额外开放 `8000/80/443`

服务架构：
- H5 页面：手机按上/中/下三层扇形引导采集图片（每层从左到右 8 个视角格，共 24 格），并在界面显示 3D 人头 Mesh 覆盖；每格自动保留 Top-K 图片
- FastAPI：任务创建、状态/日志/结果查询
- Redis + RQ：异步任务队列
- Worker：调用 `2DGSPipe/run.py`

## 1. 连接服务器

如果本机已配置 SSH 别名：

```sshconfig
Host 服务器2
    HostName 166.111.86.40
    Port 8107
    User lichengkai
```

可直接登录：

```bash
ssh 服务器2
```

或不使用别名：

```bash
ssh -p 8107 lichengkai@166.111.86.40
```

## 2. 安装服务依赖

```bash
cd /home/lichengkai/RGB_Recon
/home/lichengkai/anaconda3/envs/2DGSPipe/bin/pip install -r service/requirements.txt
```

默认复用 `2DGSPipe` conda 环境运行 API/Worker。

## 3. 环境变量（建议最小配置）

可以不配 `.env`，代码有默认值；但建议至少建一个：

```bash
cd /home/lichengkai/RGB_Recon
cp service/.env.example service/.env
```

建议在 `service/.env` 保留/确认这些项：

```bash
REDIS_URL=redis://127.0.0.1:6379/0
RQ_QUEUE_NAME=rgb_recon
PIPELINE_PYTHON=/home/lichengkai/anaconda3/envs/2DGSPipe/bin/python
BLENDER5_BIN=/home/lichengkai/RGB_Recon/2DGSPipe/uvexport/blender-5.0.1-linux-x64/blender
```

说明：
- Worker 会强制以 `--gpu auto` 启动流水线，并在进程环境里移除继承的 `CUDA_VISIBLE_DEVICES`，由 `2DGSPipe/run.py` 自动选卡。

## 4. 启动 Redis

```bash
redis-server --daemonize yes
```

如果提示 `command not found`，先安装 Redis（系统层依赖）。

## 5. 启动 API 与 Worker（仅本机监听）

建议 API 绑定 `127.0.0.1`，通过 SSH 隧道访问：

终端 A（API）：

```bash
cd /home/lichengkai/RGB_Recon
set -a && source service/.env && set +a
/home/lichengkai/anaconda3/envs/2DGSPipe/bin/python -m uvicorn service.api.app:app --host 127.0.0.1 --port 8000
```

终端 B（Worker）：

```bash
cd /home/lichengkai/RGB_Recon
set -a && source service/.env && set +a
/home/lichengkai/anaconda3/envs/2DGSPipe/bin/rq worker --url "$REDIS_URL" "$RQ_QUEUE_NAME"
```

## 6. 从本机访问（不开放新端口）

在你本地电脑执行：

```bash
ssh -p 8107 -N -L 18000:127.0.0.1:8000 lichengkai@166.111.86.40
```

然后本地浏览器打开：

```text
http://127.0.0.1:18000/
```

## 7. 手机访问（在“只开 SSH 端口”前提下）

SSH 本地转发一般不能直接给手机用。手机要访问，推荐两种方式：

1. 用 `cloudflared` 暴露 HTTPS 地址（不需要开放服务器新端口）  
2. 让运维开放 `443`，再配 Nginx 反代（长期方案）

临时方案示例（在服务器上）：

```bash
~/bin/cloudflared tunnel --url http://127.0.0.1:8000
```

命令会返回一个 `https://xxxx.trycloudflare.com`，手机访问该地址即可。

## 8. 接口清单

- `POST /api/jobs/images`（上传多张图片，后端从 `mat-face-recon-uv-tex` 开始）
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/logs`
- `GET /api/jobs/{job_id}/result`

## 9. systemd / Nginx 参考

- `service/deploy/rgb-recon-api.service`
- `service/deploy/rgb-recon-worker.service`
- `service/deploy/nginx-rgb-recon.conf`

如果后续开放 `443`，可切换到 Nginx + HTTPS 正式部署。
