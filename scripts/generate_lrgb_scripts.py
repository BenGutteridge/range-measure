"""
Generates bash scripts and/or slurm scripts for running LRGB experiments.
You may have to change slurm commands and other params based on your configuration.
"""

# =============================================================================
# User Configurable Parameters
# =============================================================================

# List of configuration file paths.
cfg_paths = [
    ## Corresponds to Figure 6 in the paper
    "configs/LRGB-tuned/vocsuperpixels-GatedGCN.yaml",
    "configs/LRGB-tuned/vocsuperpixels-GINE.yaml",
    "configs/LRGB-tuned/vocsuperpixels-GCN.yaml",
    "configs/LRGB-tuned/vocsuperpixels-GPS.yaml",
    ## Corresponds to Figure 7 in the paper
    "configs/LRGB-tuned/peptides-func-GCN.yaml",
    "configs/LRGB-tuned/peptides-func-GINE.yaml",
    "configs/LRGB-tuned/peptides-func-GatedGCN.yaml",
    "configs/LRGB-tuned/peptides-func-GPS.yaml",
    "configs/LRGB-tuned/peptides-struct-GCN.yaml",
    "configs/LRGB-tuned/peptides-struct-GINE.yaml",
    "configs/LRGB-tuned/peptides-struct-GatedGCN.yaml",
    "configs/LRGB-tuned/peptides-struct-GPS.yaml",
]

# Experiment hyperparameters
seeds = [0]  # Example seed values
split_strings = ["val"]  # Data splits to use
qos = "medium"  # Quality-of-service level (short, medium, long)
every_n_epochs = 20  # Step size for selecting epochs
num_parallel = 8  # Number of parallel commands per generated script

dataset_total_epochs = {
    "vocsuperpixels": 200,
    "peptides-func": 250,
    "peptides-struct": 250,
}  # total number of epochs for each dataset
dataset_subsets = {
    "vocsuperpixels": [500],
    "peptides-func": [200],
    "peptides-struct": [200],
}  # number of graphs to evaluate per split. Use -1 for full set

# Directories for outputs (change these if needed)
run_dir = "results/sota"  # Output directory for experiment results
scripts_output_dir = "scripts/slurm"  # Directory for generated slurm scripts
logs_base_dir = "slurm_logs/sota_ranges"  # Base directory for log files
bash_scripts_dir = "scripts/bash"  # Directory for plain bash scripts


# =============================================================================
# Internal Constants, Mappings, and Templates
# =============================================================================

# Mapping dictionaries for abbreviations (less likely to change)
dataset_mapping = {"vocsuperpixels": "V", "peptides-func": "F", "peptides-struct": "S"}
model_mapping = {"GCN": "G", "GatedGCN": "GG", "GINE": "GI", "GPS": "GP"}
split_mapping = {"train": "tr", "val": "va", "test": "te"}

# Compute short qos code str.
qos_short = {"short": "sh", "medium": "me", "long": "lo"}[qos]

# Slurm Script Template (if using slurm)
slurm_template = """#!/bin/bash
#SBATCH --job-name={qos_short}{job_name}
#SBATCH --output={log_dir}/%j.out
#SBATCH --qos={qos}
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=16
# #SBATCH --partition=h100

source /slurm-storage/miniconda3/etc/profile.d/conda.sh
conda activate longrange
cd lrgb_exps

{commands}

wait
"""


# =============================================================================
# Helper Functions
# =============================================================================
import itertools
import math
from collections import defaultdict


def sanitize(string):
    """Sanitize a string to include only alphanumerics, hyphens, and underscores."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in string)


def parse_cfg_filename(cfg_path):
    """
    Extract dataset and model names from the configuration filename.
    E.g., "vocsuperpixels-GCN.yaml" -> ("vocsuperpixels", "GCN")
    """
    filename = os.path.splitext(os.path.basename(cfg_path))[0]
    parts = filename.rsplit("-", 1)
    if len(parts) == 2:
        dataset, model = parts
    else:
        dataset = parts[0]
        model = "UnknownModel"
    return dataset, model


def map_abbreviations(dataset, model, split, subset, seed):
    """
    Map full names to their respective abbreviations for a job name.
    """
    dataset_abbr = dataset_mapping.get(dataset, "U")
    if dataset_abbr == "U":
        print(
            f"Warning: Dataset '{dataset}' not recognized. Using 'U' as abbreviation."
        )
    model_abbr = model_mapping.get(model, "U")
    if model_abbr == "U":
        print(f"Warning: Model '{model}' not recognized. Using 'U' as abbreviation.")
    split_abbr = split_mapping.get(split, "U")
    if split_abbr == "U":
        print(f"Warning: Split '{split}' not recognized. Using 'U' as abbreviation.")
    subset_abbr = f"S{subset}" if subset != -1 else ""
    seed_abbr = f"s{seed}"
    job_name = f"{dataset_abbr}{model_abbr}{split_abbr}"
    if subset_abbr:
        job_name += subset_abbr
    job_name += seed_abbr
    return job_name


def get_command(cfg, seed, subset, split, epoch, log_dir, run_dir):
    """Return a command string for a given epoch."""
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
        f"    > ../{log_dir}/epoch{epoch}.out 2>&1 &\n"
    )
    return cmd


# =============================================================================
# Job Generation (Common)
# =============================================================================
def generate_jobs():
    """
    Build a list of job dictionaries. Each job corresponds to a chunk of epochs
    (up to num_parallel commands) for a given (cfg, seed, subset, split) grouping.
    """
    tasks = []
    # Generate individual tasks for each combination.
    for cfg, seed, split in itertools.product(cfg_paths, seeds, split_strings):
        dataset, model = parse_cfg_filename(cfg)
        subsets = dataset_subsets[dataset]
        for subset in subsets:
            n_epochs = dataset_total_epochs.get(dataset, 200)
            if dataset not in dataset_total_epochs:
                print(
                    f"Warning: n_epochs for dataset '{dataset}' not defined. Using default n_epochs=200."
                )
            # Select epochs: initial ones, steps of every_n_epochs, and the final epoch.
            task_epochs = set([0, 1, 4])
            task_epochs.update(range(0, n_epochs, every_n_epochs))
            task_epochs.add(n_epochs - 1)
            sorted_epochs = sorted(task_epochs)
            for epoch in sorted_epochs:
                tasks.append((cfg, seed, subset, split, epoch, dataset, model))

    # Sort tasks to group them nicely.
    tasks_sorted = sorted(
        tasks,
        key=lambda s: (
            s[3],  # split
            model_mapping.get(s[6], ""),  # model (s[6])
            dataset_mapping.get(s[5], ""),  # dataset (s[5])
            s[0],  # cfg_path
            s[1],  # seed
            s[2],  # subset
            s[4],  # epoch
        ),
    )

    # Group tasks by (cfg, seed, subset, split)
    grouped_tasks = defaultdict(list)
    for task in tasks_sorted:
        cfg, seed, subset, split, epoch, dataset, model = task
        key = (cfg, seed, subset, split)
        grouped_tasks[key].append(epoch)

    # Create job entries by splitting epochs into chunks of size num_parallel.
    jobs = []
    for key, epochs in grouped_tasks.items():
        cfg, seed, subset, split = key
        dataset, model = parse_cfg_filename(cfg)
        # Create a base job name.
        job_base_name = map_abbreviations(dataset, model, split, subset, seed)
        job_base_name = sanitize(job_base_name)
        epochs_sorted = sorted(epochs)
        num_chunks = math.ceil(len(epochs_sorted) / num_parallel)
        for i in range(num_chunks):
            chunk_epochs = epochs_sorted[i * num_parallel : (i + 1) * num_parallel]
            # Create a unique job name using a hyphen-separated list of epochs.
            epoch_str = "e" + "-".join(f"{epoch:03d}" for epoch in chunk_epochs)
            job_name = f"{job_base_name}_{epoch_str}"
            job_name = sanitize(job_name)
            # Define the log directory.
            log_dir = os.path.join(logs_base_dir, job_name)
            os.makedirs(log_dir, exist_ok=True)
            job = {
                "job_name": job_name,
                "cfg": cfg,
                "seed": seed,
                "subset": subset,
                "split": split,
                "dataset": dataset,
                "model": model,
                "chunk_epochs": chunk_epochs,
                "log_dir": log_dir,
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
    commands (in parallel) and then call wait.
    """
    jobs = generate_jobs()
    print("Bash scripts generated:")
    for job in jobs:
        job_name = job["job_name"]
        commands = ""
        for epoch in job["chunk_epochs"]:
            commands += get_command(
                job["cfg"],
                job["seed"],
                job["subset"],
                job["split"],
                epoch,
                job["log_dir"],
                run_dir,
            )
        # Add a wait after launching the background tasks.
        commands += "wait\n"
        script_content = "#!/bin/bash\n" + commands
        bash_script_filename = f"{job_name}.sh"
        bash_script_path = os.path.join(bash_scripts_dir, bash_script_filename)
        with open(bash_script_path, "w") as f:
            f.write(script_content)
        # Make the bash script executable.
        os.chmod(bash_script_path, 0o755)
        print(f"bash {bash_script_path}")
        # Save the path so that the slurm script can call it.
        job["bash_script_path"] = os.path.abspath(bash_script_path)
    return jobs


# =============================================================================
# Part 2: Generate slurm Scripts that Run the Bash Scripts
# =============================================================================
def generate_slurm_scripts(jobs):
    """
    For each job (which already has an associated bash script) generate a slurm
    script that calls the bash script.
    """
    print("Slurm scripts generated:")
    for job in jobs:
        job_name = job["job_name"]
        # The commands section simply calls the generated bash script.
        slurm_commands = f"bash {job['bash_script_path']}\nwait\n"
        slurm_script_content = slurm_template.format(
            job_name=job_name,
            log_dir=job["log_dir"],
            qos=qos,
            qos_short=qos_short,
            commands=slurm_commands,
        )
        slurm_script_filename = f"{job_name}.sh"
        slurm_script_path = os.path.join(scripts_output_dir, slurm_script_filename)
        with open(slurm_script_path, "w") as f:
            f.write(slurm_script_content)
        print(f"sbatch {slurm_script_path}")


# =============================================================================
# Main Execution
# =============================================================================
import argparse
import shutil
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate bash scripts and optionally slurm scripts."
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

    # Make relevant dirs
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
