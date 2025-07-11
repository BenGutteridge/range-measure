#!/bin/bash
#SBATCH --job-name=retrain_pept_small
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=16
#SBATCH --array=0

source /slurm-storage/miniconda3/etc/profile.d/conda.sh
conda activate longrange
cd lrgb_exps

# All code for GCN subsampling analysis on a 10 graph subset for first and last epoch, for out/in channel subsampling together, in_node subsampling, and out_node subsampling
# Warning: Make sure the models are trained and saved before.
# To plot the curves you need to run the notebook/jac_sampling.py where you need to change the epoch, model and the boolean flags

# TRUE JACOBIAN, for both epochs
epoch=1
python main.py --cfg configs/LRGB-tuned/peptides-func-GCN.yaml out_dir results/debug train.auto_resume True train.batch_size 1 train.mode eval_range longrange.subset 10 longrange.track_range_measure True longrange.epoch $epoch longrange.split val train.ckpt_clean False share.dim_out 21 longrange.min_nodes 0 longrange.max_nodes 0

epoch=249
python main.py --cfg configs/LRGB-tuned/peptides-func-GCN.yaml out_dir results/debug train.auto_resume True train.batch_size 1 train.mode eval_range longrange.subset 10 longrange.track_range_measure True longrange.epoch $epoch longrange.split val train.ckpt_clean False share.dim_out 21 longrange.min_nodes 0 longrange.max_nodes 0


# CHANNELS: 
# fix in_node=out_node=1
in_node_prob=1.0
out_node_prob=1.0
epoch=1
for channel_prob in $(seq 0.1 0.1 1.0); do
    in_channel_prob=$channel_prob
    out_channel_prob=$channel_prob
    python main.py --cfg configs/LRGB-tuned/peptides-func-GCN.yaml out_dir results/debug train.auto_resume True train.batch_size 1 train.mode eval_range longrange.subset 10 longrange.track_range_measure True longrange.epoch $epoch longrange.split val train.ckpt_clean False share.dim_out 21 longrange.in_node_prob $in_node_prob longrange.out_node_prob $out_node_prob longrange.in_channel_prob $in_channel_prob longrange.out_channel_prob $out_channel_prob
done

# fix in_node=out_node=1
in_node_prob=1.0
out_node_prob=1.0
epoch=249
for channel_prob in $(seq 0.1 0.1 1.0); do
    in_channel_prob=$channel_prob
    out_channel_prob=$channel_prob
    python main.py --cfg configs/LRGB-tuned/peptides-func-GCN.yaml out_dir results/debug train.auto_resume True train.batch_size 1 train.mode eval_range longrange.subset 10 longrange.track_range_measure True longrange.epoch $epoch longrange.split val train.ckpt_clean False share.dim_out 21 longrange.in_node_prob $in_node_prob longrange.out_node_prob $out_node_prob longrange.in_channel_prob $in_channel_prob longrange.out_channel_prob $out_channel_prob
done

# IN NODES
in_channel_prob=1.0
out_channel_prob=1.0
out_node_prob=1.0
epoch=1
# fix in_channel=out_channel=out_node=1
for in_node_prob in $(seq 0.1 0.1 1.0); do
    python main.py --cfg configs/LRGB-tuned/peptides-func-GCN.yaml out_dir results/debug train.auto_resume True train.batch_size 1 train.mode eval_range longrange.subset 10 longrange.track_range_measure True longrange.epoch $epoch longrange.split val train.ckpt_clean False share.dim_out 21 longrange.in_node_prob $in_node_prob longrange.out_node_prob $out_node_prob longrange.in_channel_prob $in_channel_prob longrange.out_channel_prob $out_channel_prob
done

in_channel_prob=1.0
out_channel_prob=1.0
out_node_prob=1.0
epoch=249
# fix in_channel=out_channel=out_node=1
for in_node_prob in $(seq 0.1 0.1 1.0); do
    python main.py --cfg configs/LRGB-tuned/peptides-func-GCN.yaml out_dir results/debug train.auto_resume True train.batch_size 1 train.mode eval_range longrange.subset 10 longrange.track_range_measure True longrange.epoch $epoch longrange.split val train.ckpt_clean False share.dim_out 21 longrange.in_node_prob $in_node_prob longrange.out_node_prob $out_node_prob longrange.in_channel_prob $in_channel_prob longrange.out_channel_prob $out_channel_prob
done


# OUT NODES
in_channel_prob=1.0
out_channel_prob=1.0
in_node_prob=1.0
# fix in_channel=out_channel=out_node=1
epoch=1
for out_node_prob in $(seq 0.1 0.1 1.0); do
    python main.py --cfg configs/LRGB-tuned/peptides-func-GCN.yaml out_dir results/debug train.auto_resume True train.batch_size 1 train.mode eval_range longrange.subset 10 longrange.track_range_measure True longrange.epoch $epoch longrange.split val train.ckpt_clean False share.dim_out 21 longrange.in_node_prob $in_node_prob longrange.out_node_prob $out_node_prob longrange.in_channel_prob $in_channel_prob longrange.out_channel_prob $out_channel_prob
done

in_channel_prob=1.0
out_channel_prob=1.0
in_node_prob=1.0
# fix in_channel=out_channel=out_node=1
epoch=249
for out_node_prob in $(seq 0.1 0.1 1.0); do
    python main.py --cfg configs/LRGB-tuned/peptides-func-GCN.yaml out_dir results/debug train.auto_resume True train.batch_size 1 train.mode eval_range longrange.subset 10 longrange.track_range_measure True longrange.epoch $epoch longrange.split val train.ckpt_clean False share.dim_out 21 longrange.in_node_prob $in_node_prob longrange.out_node_prob $out_node_prob longrange.in_channel_prob $in_channel_prob longrange.out_channel_prob $out_channel_prob
done


