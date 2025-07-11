#!/bin/bash
#SBATCH --job-name=retrain_pept_small
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=16
#SBATCH --array=1-23

source /slurm-storage/miniconda3/etc/profile.d/conda.sh
conda activate longrange
cd lrgb_exps

paths=(
    "configs/LRGB-tuned/peptides-func-GCN.yaml"
    "configs/LRGB-tuned/peptides-func-GatedGCN.yaml"
    "configs/LRGB-tuned/peptides-func-GINE.yaml"
    "configs/LRGB-tuned/peptides-func-GPS.yaml"
    "configs/LRGB-tuned/peptides-struct-GatedGCN.yaml"
    "configs/LRGB-tuned/peptides-struct-GCN.yaml"
    "configs/LRGB-tuned/peptides-struct-GINE.yaml"
    "configs/LRGB-tuned/peptides-struct-GPS.yaml"
)

# Compute config index and seed from SLURM_ARRAY_TASK_ID.
# There are 3 seeds (0, 1, 2) per config.
task_id=$SLURM_ARRAY_TASK_ID
cfg_index=$(( task_id / 3))
seed=$(( task_id % 3))

cfg="${paths[$cfg_index]}"
echo "Running config: $cfg with seeds 0, 1, 2:"

python main.py --cfg "$cfg" longrange.track_range_measure True train.ckpt_period 1 train.ckpt_clean False out_dir results/sota seed $seed train.auto_resume True

#  & 
#     python main.py --cfg "$cfg" longrange.track_range_measure True train.ckpt_period 1 train.ckpt_clean False out_dir results/sota seed 1 &
#     python main.py --cfg "$cfg" longrange.track_range_measure True train.ckpt_period 1 train.ckpt_clean False out_dir results/sota seed 2

# wait