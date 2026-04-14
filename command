nohup /home/lichengkai/anaconda3/envs/2DGSPipe/bin/python /home/lichengkai/RGB_Recon/2DGSPipe/run_batch_selected.py \
--save_roots /home/lichengkai/RGB_Recon/service_data/jobs/4110bb9cda36458f8e36f9b443314e86 \
/home/lichengkai/RGB_Recon/service_data/jobs/4110bb9cda36458f8e36f9b443314e86/work \
/home/lichengkai/RGB_Recon/service_data/jobs/36881e1d54904f9fbe177abd1997b4c6 \
/home/lichengkai/RGB_Recon/service_data/jobs/ec931203d2e8486d8b10008d06e33ac3 \
--func uv-tex \
--gpu auto \
--max_image_side 1280 \
--continue_on_error > /home/lichengkai/RGB_Recon/rerun_jobs_$(date +%F_%H%M%S).log 2>&1 &