#!/bin/bash
#SBATCH --job-name=funVNgcn
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=15
#SBATCH --array=0
# #SBATCH --partition=h100

source /slurm-storage/miniconda3/etc/profile.d/conda.sh
conda activate longrange_copy
cd lrgb_exps

# Determine model and seed based on SLURM_ARRAY_TASK_ID.
if [ "$SLURM_ARRAY_TASK_ID" -lt 5 ]; then
    model="GCN"
    sampling_seed="$SLURM_ARRAY_TASK_ID"
else
    model="GPS"
    sampling_seed=$(($SLURM_ARRAY_TASK_ID - 5))
fi

# Optionally print the parameters for verification.
echo "Running job with model ${model} and seed ${seed}"

task="peptides-func"
epochs=(1 249)
subset=10
min_nodes=16
max_nodes=256
run_seed=0
vn="True"

# Optional: train the model
python main.py \
        --cfg "configs/LRGB-tuned/${task}-${model}.yaml" \
        longrange.track_range_measure True \
        train.ckpt_period 1 \
        train.ckpt_clean False \
        train.auto_resume True \
        out_dir results/VN \
        seed "$run_seed" \
        dataset.virtual_node $vn


# # TRUE JACOBIAN, for both epochs
# for epoch in "${epochs[@]}"; do
#     python main.py \
#         --cfg "configs/LRGB-tuned/${task}-${model}.yaml" \
#         out_dir results/sampling \
#         train.auto_resume True \
#         seed "$run_seed" \
#         train.batch_size 1 \
#         train.mode eval_range \
#         longrange.subset "$subset" \
#         longrange.track_range_measure True \
#         longrange.epoch "$epoch" \
#         longrange.split val \
#         train.ckpt_clean False \
#         longrange.use_sampled_jacobian False
# done

# # CHANNELS:
# # fix in_node=out_nodes=1, vary in_channels=out_channels
# in_node_prob=1.0
# out_node_prob=1.0

# for epoch in "${epochs[@]}"; do
#     for channel_prob in $(seq 0.1 0.1 1.0); do
#         in_channel_prob="$channel_prob"
#         out_channel_prob="$channel_prob"
#         python main.py \
#             --cfg "configs/LRGB-tuned/${task}-${model}.yaml" \
#             out_dir results/sampling \
#             train.auto_resume True \
#             seed "$run_seed" \
#             train.batch_size 1 \
#             train.mode eval_range \
#             longrange.subset "$subset" \
#             longrange.track_range_measure True \
#             longrange.epoch "$epoch" \
#             longrange.split val \
#             train.ckpt_clean False \
#             longrange.in_node_prob "$in_node_prob" \
#             longrange.out_node_prob "$out_node_prob" \
#             longrange.in_channel_prob "$in_channel_prob" \
#             longrange.out_channel_prob "$out_channel_prob" \
#             longrange.min_nodes "$min_nodes" \
#             longrange.max_nodes "$max_nodes" \
#             longrange.sampling_seed "$sampling_seed"
#     done
# done

# # IN NODES:
# # fix in_channels=out_channels=out_nodes=1, vary in_nodes
# in_channel_prob=1.0
# out_channel_prob=1.0
# out_node_prob=1.0

# for epoch in "${epochs[@]}"; do
#     for in_node_prob in $(seq 0.1 0.1 1.0); do
#         python main.py \
#             --cfg "configs/LRGB-tuned/${task}-${model}.yaml" \
#             out_dir results/sampling \
#             train.auto_resume True \
#             seed "$run_seed" \
#             train.batch_size 1 \
#             train.mode eval_range \
#             longrange.subset "$subset" \
#             longrange.track_range_measure True \
#             longrange.epoch "$epoch" \
#             longrange.split val \
#             train.ckpt_clean False \
#             longrange.in_node_prob "$in_node_prob" \
#             longrange.out_node_prob "$out_node_prob" \
#             longrange.in_channel_prob "$in_channel_prob" \
#             longrange.out_channel_prob "$out_channel_prob" \
#             longrange.min_nodes "$min_nodes" \
#             longrange.max_nodes "$max_nodes" \
#             longrange.sampling_seed "$sampling_seed"
#     done
# done

# # OUT NODES:
# # fix in_channels=out_channels=in_nodes=1, vary out_nodes
# in_channel_prob=1.0
# out_channel_prob=1.0
# in_node_prob=1.0

# for epoch in "${epochs[@]}"; do
#     for out_node_prob in $(seq 0.1 0.1 1.0); do
#         python main.py \
#             --cfg "configs/LRGB-tuned/${task}-${model}.yaml" \
#             out_dir results/sampling \
#             train.auto_resume True \
#             seed "$run_seed" \
#             train.batch_size 1 \
#             train.mode eval_range \
#             longrange.subset "$subset" \
#             longrange.track_range_measure True \
#             longrange.epoch "$epoch" \
#             longrange.split val \
#             train.ckpt_clean False \
#             longrange.in_node_prob "$in_node_prob" \
#             longrange.out_node_prob "$out_node_prob" \
#             longrange.in_channel_prob "$in_channel_prob" \
#             longrange.out_channel_prob "$out_channel_prob" \
#             longrange.min_nodes "$min_nodes" \
#             longrange.max_nodes "$max_nodes" \
#             longrange.sampling_seed "$sampling_seed"
#     done
# done
