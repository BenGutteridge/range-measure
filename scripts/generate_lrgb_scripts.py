"""
Updated script for generating bash/slurm scripts for LRGB experiments.
Now, commands across all seeds for a given config (and subset/split) are merged
and chunked into parallel groups of size num_parallel.
"""

import argparse
import shutil
import os
import math
from collections import defaultdict

# =============================================================================
# User Configurable Parameters
# =============================================================================

# List of configuration file paths.
cfg_paths = [
    # Corresponds to Figure 6 in the paper
    # "configs/LRGB-tuned/vocsuperpixels-GCN.yaml",
    # "configs/LRGB-tuned/vocsuperpixels-GatedGCN.yaml",
    # "configs/LRGB-tuned/vocsuperpixels-GINE.yaml",
    "configs/LRGB-tuned/vocsuperpixels-GPS.yaml",
    "configs/LRGB-tuned/vocsuperpixels-Transformer.yaml",
    # "configs/LRGB-tuned/vocsuperpixels-GCN+VN.yaml",
    # Corresponds to Figure 7 in the paper
    # "configs/LRGB-tuned/peptides-func-GCN.yaml",
    # "configs/LRGB-tuned/peptides-func-GatedGCN.yaml",
    # "configs/LRGB-tuned/peptides-func-GINE.yaml",
    "configs/LRGB-tuned/peptides-func-GPS.yaml",
    "configs/LRGB-tuned/peptides-func-Transformer.yaml",
    # "configs/LRGB-tuned/peptides-func-GCN+VN.yaml",
    # "configs/LRGB-tuned/peptides-struct-GCN.yaml",
    # "configs/LRGB-tuned/peptides-struct-GatedGCN.yaml",
    # "configs/LRGB-tuned/peptides-struct-GINE.yaml",
    "configs/LRGB-tuned/peptides-struct-GPS.yaml",
    "configs/LRGB-tuned/peptides-struct-Transformer.yaml",
    # "configs/LRGB-tuned/peptides-struct-GCN+VN.yaml",
]

# Experiment hyperparameters
seeds = [0, 1, 2, 3]  # All seeds will be merged across a given config.
split_strings = ["val"]  # Data splits to use
qos = "short"  # Quality-of-service level (short, medium, long)
num_parallel = 1  # Number of parallel commands per generated script
use_h100 = False

# Step size for selecting epochs (per dataset)
every_n_epochs = {
    "peptides-func": 20,
    "peptides-struct": 20,
    "vocsuperpixels": 20,
}

# Total epochs per dataset
dataset_total_epochs = {
    "peptides-func": 250,
    "peptides-struct": 250,
    "vocsuperpixels": 200,
}

# Subset sizes to evaluate (change as needed)
dataset_subsets = {
    "peptides-func": [200],
    "peptides-struct": [200],
    "vocsuperpixels": [500],
}

# Sampling probabilities (per dataset)
sampling_probs = {
    "peptides-func": {
        "in_channel": 0.5,
        "in_node": 1.0,
        "out_channel": 0.5,
        "out_node": 0.5,
    },
    "peptides-struct": {
        "in_channel": 0.5,
        "in_node": 1.0,
        "out_channel": 0.5,
        "out_node": 0.5,
    },
    "vocsuperpixels": {
        "in_channel": 0.5,
        "in_node": 1.0,
        "out_channel": 0.5,
        "out_node": 0.5,
    },
}

# Directories for outputs (change these if needed)
exp_set_name = ""  # Name of the experiment set
run_dir = "results/camera_ready"  # Output directory for experiment results
scripts_output_dir = os.path.join("scripts/slurm", exp_set_name)  # Directory for generated slurm scripts
logs_base_dir = os.path.join("slurm_logs/camera_ready", exp_set_name)  # Base directory for log files
bash_scripts_dir = os.path.join("scripts/bash", exp_set_name)  # Directory for generated bash scripts

# =============================================================================
# Internal Constants, Mappings, and Templates
# =============================================================================

# Mapping dictionaries for abbreviations
dataset_mapping = {
    "vocsuperpixels": "V",
    "peptides-func": "F",
    "peptides-struct": "S",
    "zinc": "Z",
}
model_mapping = {
    "GCN": "GC",
    "GatedGCN": "GG",
    "GINE": "GI",
    "GPS": "GP",
    "GPS+RWSE": "GP",
    "GINE+RWSE": "GI",
    "GCN+VN": "GV",
    "Transformer": "GT",
}
split_mapping = {"train": "tr", "val": "", "test": "te"}
qos_short = {"short": "sh", "medium": "me", "long": "lo"}[qos]

# Slurm Script Template (if using slurm)
slurm_template = """#!/bin/bash
#SBATCH --job-name={qos_short}{job_name}
#SBATCH --output={log_dir}/%j.out
#SBATCH --qos={qos}
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100

source /slurm-storage/miniconda3/etc/profile.d/conda.sh
conda activate longrange_copy
cd lrgb_exps

{commands}

wait
"""
if not use_h100:
    slurm_template = slurm_template.replace("#SBATCH --partition=h100\n", "\n")

# =============================================================================
# Helper Functions
# =============================================================================

def sanitize(string):
    """Sanitize a string to include only alphanumerics, hyphens, and underscores."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in string)

def parse_cfg_filename(cfg_path):
    """
    Extract dataset and model names from the configuration filename.
    E.g., "peptides-func-GCN.yaml" -> ("peptides-func", "GCN")
    """
    filename = os.path.splitext(os.path.basename(cfg_path))[0]
    parts = filename.rsplit("-", 1)
    if len(parts) == 2:
        dataset, model = parts
    else:
        dataset = parts[0]
        model = "UnknownModel"
    return dataset, model

def map_abbreviations_no_seed(dataset, model, split):
    """
    Create a job name abbreviation that does not include seed information.
    """
    dataset_abbr = dataset_mapping.get(dataset, "U")
    if dataset_abbr == "U":
        print(f"Warning: Dataset '{dataset}' not recognized. Using 'U' as abbreviation.")
    model_abbr = model_mapping.get(model, "U")
    if model_abbr == "U":
        print(f"Warning: Model '{model}' not recognized. Using 'U' as abbreviation.")
    split_abbr = split_mapping.get(split, "U")
    if split_abbr == "U":
        print(f"Warning: Split '{split}' not recognized. Using 'U' as abbreviation.")
    job_name = f"{dataset_abbr}{model_abbr}{split_abbr}"
    return job_name

def get_command(cfg, seed, subset, split, epoch, log_dir, run_dir, sampling_probs):
    """Return a command string for a given (seed, epoch) combination."""
    cmd = (
        f'python main.py --cfg "{cfg}" \\\n'
        f"    longrange.track_range_measure True \\\n"
        f"    train.ckpt_period 1 \\\n"
        f"    train.ckpt_clean False \\\n"
        f"    out_dir {run_dir} \\\n"
        f"    seed {seed} \\\n"
        f"    train.auto_resume True \\\n"
        f"    train.mode eval_range \\\n"
        f"    train.batch_size 1 \\\n"
        f"    longrange.subset {subset} \\\n"
        f"    longrange.split {split} \\\n"
        f"    longrange.epoch {epoch} \\\n"
        f"    longrange.in_channel_prob {sampling_probs['in_channel']} \\\n"
        f"    longrange.in_node_prob {sampling_probs['in_node']} \\\n"
        f"    longrange.out_channel_prob {sampling_probs['out_channel']} \\\n"
        f"    longrange.out_node_prob {sampling_probs['out_node']} \\\n"
        f"    longrange.sampling_seed {seed} \\\n"
        f'    > "../{log_dir}/${{SLURM_JOB_ID}}_e{epoch:03d}.out" 2>&1 &\n'
    )
    return cmd

# =============================================================================
# Job Generation (Modified to Parallelize Across Seeds)
# =============================================================================
def generate_jobs():
    """
    Build a list of job dictionaries. Each job now corresponds to a chunk of commands
    (up to num_parallel commands) for a given (cfg, subset, split) grouping, where the commands
    span all seeds and their respective epochs.
    """
    tasks_by_config = defaultdict(list)
    # Generate tasks for each configuration grouping (cfg, subset, split)
    for cfg in cfg_paths:
        dataset, model = parse_cfg_filename(cfg)
        subsets = dataset_subsets[dataset]
        for split in split_strings:
            for subset in subsets:
                for seed in seeds:
                    n_epochs = dataset_total_epochs.get(dataset, 200)
                    if dataset not in dataset_total_epochs:
                        print(f"Warning: n_epochs for dataset '{dataset}' not defined. Using default n_epochs=200.")
                    # Select epochs: initial ones, steps of every_n_epochs, and the final epoch.
                    task_epochs = set([-1, 0, 1, 2, 4])
                    task_epochs.update(range(0, n_epochs, every_n_epochs.get(dataset, 20)))
                    task_epochs.add(n_epochs - 1)
                    sorted_epochs = sorted(task_epochs)
                    for epoch in sorted_epochs:                       
                        range_stats_path = (
                            f"lrgb_exps/{run_dir}/{dataset}-{model}/{seed}/{split}/range_stats_subset{subset}_samples"
                            f"-channels-in{sampling_probs[dataset]['in_channel']:.2f}out{sampling_probs[dataset]['out_channel']:.2f}-nodes-in{sampling_probs[dataset]['in_node']:.2f}out{sampling_probs[dataset]['out_node']:.2f}"
                            f"-min16-max256_seed{seed}_e{epoch:03d}.csv"
                        )
                        # Don't bother for tasks that have been completed already
                        if not os.path.exists(range_stats_path):
                            tasks_by_config[(cfg, subset, split)].append((seed, epoch))
    
    jobs = []
    # For each configuration grouping, chunk the tasks into groups of num_parallel commands.
    for (cfg, subset, split), task_list in tasks_by_config.items():
        dataset, model = parse_cfg_filename(cfg)
        # Sort tasks first by seed then by epoch.
        task_list.sort(key=lambda x: (x[0], x[1]))
        num_chunks = math.ceil(len(task_list) / num_parallel)
        base_job_name = sanitize(map_abbreviations_no_seed(dataset, model, split))
        for i in range(num_chunks):
            chunk_tasks = task_list[i * num_parallel : (i + 1) * num_parallel]
            # For naming, we include only the epoch numbers from this chunk (ignoring seed).
            epochs_in_chunk = [t[1] for t in chunk_tasks]
            epoch_str = "e" + "-".join(f"{epoch}" for epoch in epochs_in_chunk)
            job_name = sanitize(f"{base_job_name}{epoch_str}")
            job = {
                "job_name": job_name,
                "cfg": cfg,
                "subset": subset,
                "split": split,
                "dataset": dataset,
                "model": model,
                "chunk_tasks": chunk_tasks,  # list of (seed, epoch) tuples
                "log_dir": logs_base_dir,
            }
            jobs.append(job)
    return jobs

# =============================================================================
# Part 1: Generate Plain Bash Scripts (Non-slurm)
# =============================================================================
def generate_bash_scripts():
    """
    For each job, generate a plain bash script (without any slurm headers)
    that runs the desired commands. Each bash script will run up to num_parallel
    commands (from possibly multiple seeds) in parallel and then call wait.
    """
    jobs = generate_jobs()
    print("Bash scripts generated:")
    for job in jobs:
        job_name = job["job_name"]
        commands = ""
        seed_epoch_strs = [] 
        for (seed, epoch) in job["chunk_tasks"]:
            commands += get_command(
                cfg=job["cfg"],
                seed=seed,
                subset=job["subset"],
                split=job["split"],
                epoch=epoch,
                log_dir=job["log_dir"],
                run_dir=run_dir,
                sampling_probs=sampling_probs[job["dataset"]],
            )
            seed_epoch_strs.append(f"s{seed}e{epoch:03d}")
        # Ensure all background processes complete
        commands += "wait\n"
        script_content = "#!/bin/bash\n\nSLURM_JOB_ID=$1\n" + commands
        
        bash_script_filename = f"{job_name.split('e')[0]}{'-'.join(seed_epoch_strs)}.sh"
        bash_script_path = os.path.join(bash_scripts_dir, bash_script_filename)
        with open(bash_script_path, "w") as f:
            f.write(script_content)
        os.chmod(bash_script_path, 0o755)
        print(f"bash {bash_script_path}")
        # Store the absolute path for slurm scripts to reference.
        job["bash_script_path"] = os.path.abspath(bash_script_path)
    return jobs

# =============================================================================
# Part 2: Generate Slurm Scripts that Run the Bash Scripts
# =============================================================================
def generate_slurm_scripts(jobs):
    """
    For each job (which already has an associated bash script) generate a slurm
    script that calls the bash script.
    """
    print("Slurm scripts generated:")
    for job in jobs:
        job_name = job["job_name"]
        # The slurm commands call the corresponding bash script.
        slurm_commands = f"bash {job['bash_script_path']} $SLURM_JOB_ID \nwait\n"
        slurm_script_content = slurm_template.format(
            job_name=job_name,
            log_dir=job["log_dir"],
            qos=qos,
            qos_short=qos_short,
            commands=slurm_commands,
        )
        slurm_script_filename = job["bash_script_path"].split("/")[-1]
        slurm_script_path = os.path.join(scripts_output_dir, slurm_script_filename)
        with open(slurm_script_path, "w") as f:
            f.write(slurm_script_content)
        print(f"sbatch {slurm_script_path}")

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate bash scripts and optionally slurm scripts for LRGB experiments."
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Generate slurm scripts that call the bash scripts.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing script dirs.",
    )
    args = parser.parse_args()

    # Create/clean necessary directories.
    if args.clean:
        if os.path.exists(bash_scripts_dir):
            print("\nOverwriting existing bash script dir...")
            shutil.rmtree(bash_scripts_dir)
        if os.path.exists(scripts_output_dir) and args.slurm:
            print("\nOverwriting existing slurm script dir...")
            shutil.rmtree(scripts_output_dir)
    os.makedirs(bash_scripts_dir, exist_ok=True)
    os.makedirs(logs_base_dir, exist_ok=True)
    if args.slurm:
        os.makedirs(scripts_output_dir, exist_ok=True)

    jobs = generate_bash_scripts()
    print("\nBash scripts have been generated.")

    if args.slurm:
        print("\nGenerating slurm scripts that call the bash scripts...")
        generate_slurm_scripts(jobs)
        print("\nAll scripts have been generated.")
    else:
        print("\nSkipping slurm script generation.")
