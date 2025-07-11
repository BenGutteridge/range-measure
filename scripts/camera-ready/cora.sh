#!/bin/bash
#SBATCH --job-name=densecora
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=15

source /slurm-storage/miniconda3/etc/profile.d/conda.sh
conda activate longrange_copy

model="GCN"

python main.py \
  --dataset cora \
  --model_type $model \
  --skip \
  --hidden_dim 64 \
  --num_layers 2 \
  --seed 0 \
  --max_epochs 200 \
  --lr 0.01 \
  --track_epoch_every 25 \
  --track_epochs 1 2 3 4 9 14 19 \
  --gpu 0 \
  --track_range \
  --out_channel_sample_prob 1.0 \
  --in_channel_sample_prob 0.2 \
  --in_node_sample_prob 1.0 \
  --out_node_sample_prob 0.2

model="GT_dense"

python main.py \
  --dataset cora \
  --model_type $model \
  --skip \
  --hidden_dim 64 \
  --num_layers 2 \
  --seed 0 \
  --max_epochs 200 \
  --lr 0.01 \
  --track_epoch_every 25 \
  --track_epochs 1 2 3 4 9 14 19 \
  --gpu 0 \
  --out_channel_sample_prob 1.0 \
  --in_channel_sample_prob 0.2 \
  --in_node_sample_prob 1.0 \
  --out_node_sample_prob 0.2 \
  --track_range
