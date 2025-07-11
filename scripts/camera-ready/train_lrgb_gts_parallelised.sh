#!/bin/bash
#SBATCH --job-name=LRGTseeds
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=16
#SBATCH --partition=h100


source /slurm-storage/miniconda3/etc/profile.d/conda.sh
conda activate longrange_copy
cd lrgb_exps || exit 1

# ------------- Configs and seeds -------------
paths=(
    "configs/LRGB-tuned/peptides-func-Transformer.yaml"
    "configs/LRGB-tuned/peptides-func-GPS.yaml"
    "configs/LRGB-tuned/peptides-struct-GPS.yaml"
    "configs/LRGB-tuned/peptides-struct-Transformer.yaml"
    "configs/LRGB-tuned/vocsuperpixels-GPS.yaml"
    "configs/LRGB-tuned/vocsuperpixels-Transformer.yaml"
)

seeds=(0 1 2 3)

# ------------- Control parallelism -------------
num_parallel=20    # <--- set this to control number of parallel processes

# ------------- Create all (config, seed) jobs -------------
jobs=()
for cfg in "${paths[@]}"; do
    for seed in "${seeds[@]}"; do
        jobs+=("$cfg $seed")
    done
done

# ------------- Run with controlled parallelism -------------
run_job() {
    cfg="$1"
    seed="$2"
    echo "Launching: $cfg with seed $seed"
    python main.py --repeat 1 --cfg "$cfg" longrange.track_range_measure True \
        train.ckpt_period 1 train.ckpt_clean False \
        out_dir results/camera_ready seed "$seed" longrange.sampling_seed "$seed"
}

# Job loop with concurrency control
i=0
running=()

while [ $i -lt ${#jobs[@]} ]; do
    # Count current running jobs
    running=($(jobs -rp))
    num_running=${#running[@]}

    if [ "$num_running" -lt "$num_parallel" ]; then
        job=(${jobs[$i]})
        run_job "${job[0]}" "${job[1]}" &
        ((i++))
    else
        # Wait for any job to finish before checking again
        wait -n
    fi
done

# Wait for all remaining jobs
wait
echo "All jobs complete."
