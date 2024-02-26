#/bin/bash

# $1: task
# $2: gpu number
# $3: output directory
# $4: c_rate, default 0.95
# $5: particle_size, default 5

python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/$1_config.yaml \
    --gpu=$2 \
    --save_dir=$3 \
    --c_rate=$4 \
    --particle_size = $5;
