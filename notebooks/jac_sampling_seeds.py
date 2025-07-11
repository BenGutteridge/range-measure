import os
from os.path import join
import re
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np

results_dir = "lrgb_exps/results/sampling"
dataset = "peptides-func"
model = "GCN"
split = "val"
run_seed = 0
sampling_seeds = [0, 1, 2, 3]
# Use a list of epochs as before
epochs = [1, 199]
subset = 10  # set as None if unused
min_nodes, max_nodes = 16, 256  # set as None if unused

runs = []
run_dir = join(results_dir, f"{dataset}-{model}/{run_seed}", split)
for epoch in epochs:  # loop over both epochs
    checks = [
        "gpu_stats",
        ".csv",
        f"e{epoch:03d}",
        f"subset{subset}" if subset is not None else "",
        f"min{min_nodes}" if min_nodes is not None else "",
        f"max{max_nodes}" if max_nodes is not None else "",
    ]

    # Load the data for the unsampled run first
    unsampled_gpu_stats_path = join(
        run_dir, f"gpu_stats_subset{subset}_e{epoch:03d}.csv"
    )
    unsampled_gpu_stats = pd.read_csv(
        unsampled_gpu_stats_path,
        header=None,
        names=["batch", "time_ms", "net_memory", "peak_memory"],
    )
    unsampled_range_stats_path = unsampled_gpu_stats_path.replace(
        "gpu_stats", "range_stats"
    )
    unsampled_range_stats = pd.read_csv(unsampled_range_stats_path, index_col=0)

    run = pd.concat(
        [
            unsampled_gpu_stats.drop(columns=["batch"]).mean(),
            unsampled_range_stats["range"],
        ]
    )
    run["run_seed"] = run_seed
    run["epoch"] = epoch
    run["channels_in"] = -1
    run["channels_out"] = -1
    run["nodes_in"] = -1
    run["nodes_out"] = -1
    run["sampling_seed"] = -1
    runs.append(run)

    for file in os.listdir(run_dir):
        if all(check in file for check in checks):
            df = pd.read_csv(
                join(run_dir, file),
                header=None,
                names=["batch", "time_ms", "net_memory", "peak_memory"],
            )
            df = df.drop_duplicates(subset="batch", keep="last").reset_index(drop=True)

            if not df["batch"].tolist() == list(range(subset)):
                logger.warning(
                    f"Missing runs for {join(run_dir, file)},\nonly loaded {df['batch'].tolist()} for subset of {subset}"
                )
                continue
            range_stats_path = join(run_dir, file).replace("gpu_stats", "range_stats")
            range_stats = pd.read_csv(range_stats_path, index_col=0)
            run = pd.concat([df.drop(columns=["batch"]).mean(), range_stats["range"]])
            run["run_seed"] = run_seed
            run["epoch"] = epoch  # add epoch information
            # Extract sampling params (probabilities) from filename
            if match := re.search(
                pattern=r"channels-in([\d\.]+)out([\d\.]+)-nodes-in([\d\.]+)-out([\d\.]+).*_seed(\d+)",
                string=file,
            ):
                (
                    channels_in,
                    channels_out,
                    nodes_in,
                    nodes_out,
                    seed,
                ) = match.groups()
                run["channels_in"] = float(channels_in)
                run["channels_out"] = float(channels_out)
                run["nodes_in"] = float(nodes_in)
                run["nodes_out"] = float(nodes_out)
                run["sampling_seed"] = int(seed)
                runs.append(run)
            else:
                logger.error(f"Could not extract sampling params from {file}")

# Get mean and std across seeds, now grouping by epoch as well
df = pd.DataFrame(runs)
df["run_seed"] = df["run_seed"].astype(int)
df["sampling_seed"] = df["sampling_seed"].astype(int)
df["epoch"] = df["epoch"].astype(int)
df = (
    df.groupby(
        ["epoch", "channels_in", "channels_out", "nodes_in", "nodes_out", "run_seed"]
    )
    .agg(["mean", "std", set])
    .reset_index()
)
df.columns = [f"{col}_{stat}" if stat else col for col, stat in df.columns]
for _, row in df.iterrows():
    if not (
        row["sampling_seed_set"] == {-1}
        or set(sampling_seeds) <= row["sampling_seed_set"]  # subset
    ):
        raise ValueError(
            f"Missing sampling seeds for {row[['epoch', 'channels_in', 'channels_out', 'nodes_in', 'nodes_out', 'run_seed']]}: {row['sampling_seed_set']}"
        )
df = df.drop(
    columns=[col for col in df.columns if any(s in col for s in ["sum", "set", "seed"])]
)

# Extract unsampled runs (nodes/channels_in/out == -1) from the aggregated data
unsampled_df = df[
    (df["channels_in"] == -1)
    & (df["channels_out"] == -1)
    & (df["nodes_in"] == -1)
    & (df["nodes_out"] == -1)
]

# Define the subplot structure:
# Row 0: time_ms_mean
# Row 1: peak_memory_mean
# Row 2: combined: range_res_mean and range_spd_mean
# Row 3: combined: range_res_norm_mean and range_spd_norm_mean
subplot_metrics = [
    [("time_ms_mean", "time_ms_std")],
    [("peak_memory_mean", "peak_memory_std")],
    [("range_res_mean", "range_res_std"), ("range_spd_mean", "range_spd_std")],
    [
        ("range_res_norm_mean", "range_res_norm_std"),
        ("range_spd_norm_mean", "range_spd_norm_std"),
    ],
]

data = {
    # Plot for fixed nodes_in=nodes_out=1.0, vary channels_in=channels_out on x axis
    "channels_in": df[(df["nodes_in"] == 1.0) & (df["nodes_out"] == 1.0)],
    # Plot for fixed channels_in=channels_out=nodes_out=1.0, vary nodes_in on x axis
    "nodes_in": df[
        (df["channels_in"] == 1.0)
        & (df["channels_out"] == 1.0)
        & (df["nodes_out"] == 1.0)
    ],
    # Plot for fixed channels_in=channels_out=nodes_in=1.0, vary nodes_out on x axis
    "nodes_out": df[
        (df["channels_in"] == 1.0)
        & (df["channels_out"] == 1.0)
        & (df["nodes_in"] == 1.0)
    ],
}

# Generate a separate figure for each item in data
# with 2 columns (one for each epoch) and 4 rows for different stats
for varying, df_ in data.items():
    num_rows = len(subplot_metrics)
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, num_rows * 4))
    for col_idx, ep in enumerate(epochs):
        df_ep = df_[df_["epoch"] == ep]
        # Get unsampled run for this epoch (should be a single row)
        unsampled_ep = unsampled_df[unsampled_df["epoch"] == ep]
        for row_idx, metrics in enumerate(subplot_metrics):
            ax = axes[row_idx, col_idx]
            for m, m_err in metrics:
                x = df_ep[f"{varying}"]
                y = df_ep[m]
                err = df_ep[m_err]
                # Plot the sampled run line with markers and capture the line object for its color.
                line = ax.plot(x, y, "-o", label=m.replace("_mean", ""))[0]
                color = line.get_color()
                # Plot the continuous error band for the sampled run.
                ax.fill_between(x, y - err, y + err, alpha=0.3, color=color)

                # Get unsampled value and error for this metric.
                unsampled_value = unsampled_ep[m].iloc[0]
                unsampled_err = unsampled_ep[m_err].iloc[0]
                # Determine the x-range from the current subplot.
                x_min = x.min()
                x_max = x.max()
                # If there is only one metric in the row, use black dashed line.
                # Otherwise, use the corresponding color.
                dashed_color = color if len(metrics) > 1 else "black"
                # Plot a horizontal dashed line for the unsampled run.
                ax.hlines(
                    unsampled_value,
                    x_min,
                    x_max,
                    linestyles="dashed",
                    colors=dashed_color,
                    label="_nolegend_",
                )
                # Plot a faint transparent error band for the unsampled run.
                ax.fill_between(
                    [x_min, x_max],
                    [unsampled_value - unsampled_err] * 2,
                    [unsampled_value + unsampled_err] * 2,
                    alpha=0.2,
                    color=dashed_color,
                )
            ax.set_xlabel(varying)
            ax.set_ylabel(metrics[0][0].replace("_mean", ""))
            ax.legend()
    fig.suptitle(
        f"{dataset}-{model} {varying} sampling, epochs {epochs[0]} and {epochs[1]}",
        fontsize=16,
        y=0.9,
    )
    os.makedirs("plots/sampling", exist_ok=True)
    figpath = f"plots/sampling/{dataset}-{model}-{varying}-sampling-epochs-{epochs[0]:03d}-{epochs[1]:03d}.png"
    fig.savefig(figpath, bbox_inches="tight")
    logger.info(
        f"Saved plot for {dataset}-{model} {varying} sampling, epochs {epochs[0]} and {epochs[1]} to {figpath}"
    )
