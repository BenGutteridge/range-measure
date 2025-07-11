#!/bin/bash
#SBATCH --job-name=LRGTseeds
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=15
#SBATCH --array=0-23  # 6 configs × 4 seeds = 24 jobs (indices 0–23)

source /slurm-storage/miniconda3/etc/profile.d/conda.sh
conda activate longrange_copy
cd lrgb_exps

paths=(
    "configs/LRGB-tuned/peptides-func-Transformer.yaml"
    "configs/LRGB-tuned/peptides-func-GPS.yaml"
    "configs/LRGB-tuned/peptides-struct-GPS.yaml"
    "configs/LRGB-tuned/peptides-struct-Transformer.yaml"
    "configs/LRGB-tuned/vocsuperpixels-GPS.yaml"
    "configs/LRGB-tuned/vocsuperpixels-Transformer.yaml"
)

# Calculate config and seed based on Slurm array ID
config_index=$((SLURM_ARRAY_TASK_ID / 4))
seed=$((SLURM_ARRAY_TASK_ID % 4))
cfg="${paths[$config_index]}"

echo "Running config: $cfg with seed: $seed"
python main.py --repeat 1 --cfg "$cfg" longrange.track_range_measure True train.ckpt_period 1 train.ckpt_clean False out_dir results/camera_ready seed "$seed" longrange.sampling_seed "$seed"
