ckpt_dir=${ckpt_dir:-./Grounded-SAM-2/checkpoints}

python3 motionblender/preproc/sam2gui/gui_app.py  --task_json $1  --checkpoint  ${ckpt_dir}/sam2.1_hiera_large.pt \
    --model_cfg  ../sam2/configs/sam2.1/sam2.1_hiera_l.yaml --port ${port:-8891}
