#!/bin/bash
#SBATCH --job-name=LRMPseeds
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=15
#SBATCH --array=7,11

source /slurm-storage/miniconda3/etc/profile.d/conda.sh
conda activate longrange_copy
cd lrgb_exps

paths=(
    "configs/LRGB-tuned/peptides-func-GCN+VN.yaml"
    "configs/LRGB-tuned/peptides-func-GatedGCN.yaml"
    "configs/LRGB-tuned/peptides-func-GCN.yaml"
    "configs/LRGB-tuned/peptides-func-GINE.yaml"
    "configs/LRGB-tuned/peptides-struct-GatedGCN.yaml"
    "configs/LRGB-tuned/peptides-struct-GCN.yaml"
    "configs/LRGB-tuned/peptides-struct-GINE.yaml"
    "configs/LRGB-tuned/peptides-struct-GCN+VN.yaml"
    "configs/LRGB-tuned/vocsuperpixels-GatedGCN.yaml"
    "configs/LRGB-tuned/vocsuperpixels-GCN.yaml"
    "configs/LRGB-tuned/vocsuperpixels-GINE.yaml"
    "configs/LRGB-tuned/vocsuperpixels-GCN+VN.yaml"
)

cfg="${paths[$SLURM_ARRAY_TASK_ID]}"
echo "Running config: $cfg with seeds 0 to 3"

for seed in {0..3}; do
    python main.py --repeat 1 --cfg "$cfg" longrange.track_range_measure True train.ckpt_period 1 train.ckpt_clean False out_dir results/camera_ready seed "$seed" longrange.sampling_seed "$seed" &
done

wait
echo "All seeds finished for config: $cfg"
