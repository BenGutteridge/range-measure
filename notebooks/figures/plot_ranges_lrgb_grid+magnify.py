"""
Notebook for plotting LRGB range measure experiments for LRGB vocsuperpixels.

Generates Figure 6 in the paper.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import font_manager
from notebooks.figures.plot_utils import (
    get_final_epoch_stats,
    load_and_aggregate_data,
)

monospace_font = font_manager.FontProperties(family="monospace", size=15)

# ------------------ Load and aggregate data ---------------------

plot_out_dir = "plots/camera_ready"
os.makedirs(plot_out_dir, exist_ok=True)
path = "lrgb_exps/results/camera_ready"
dataset = "vocsuperpixels"
subset = 500
seeds = [0, 1, 2, 3]
logy = True  # Set log-scale y axis

sampling_kwargs = {
    "in_node_prob": 1.0,
    "in_channel_prob": 0.5,
    "out_node_prob": 0.5,
    "out_channel_prob": 0.5,
    "min_nodes": 16,
    "max_nodes": 256,
}
selected_splits = [
    "val",
]
subset = {
    "vocsuperpixels": 500,
    "peptides-func": 200,
    "peptides-struct": 200,
}[dataset]

models_shortened = {
    "GatedGCN": "GGCN",
    "GCN": "GCN",
    "GPS": "GPS",
    "GINE": "GINE",
    "Transformer": "GT",
    "GCN+VN": "GCN+VN",
}

final_epoch_stats = {
    model: get_final_epoch_stats(os.path.join(path, f"{dataset}-{model}"))
    for model in models_shortened.keys()
    if os.path.exists(os.path.join(path, f"{dataset}-{model}"))
}

task_metric = {
    "vocsuperpixels": "F1",
    "peptides-func": "AP",
    "peptides-struct": "MAE",
}[dataset]


aggregated_dfs = []
for seed in seeds:
    aggregated_dfs.append(
            load_and_aggregate_data(
            path=path,
            dataset=dataset,
            subset=subset,
            seed=seed,
            **sampling_kwargs,
        )
    )

# If a single seed, use std over *graphs* for error bars
aggregated_df = aggregated_dfs[0]
# Otherwise, drop existing var cols and compute new ones across *seeds* not graphs
if len(seeds) > 1:
    range_cols = [col for col in aggregated_dfs[0].columns if col.startswith("range") and not col.endswith("_var")]
    for i in range(len(aggregated_dfs)):
        for col in range_cols: 
            # Overwrite per-graph variance with per-seed variance
            aggregated_dfs[i][f"{col}_var"] = aggregated_dfs[i][col] ** 2
    
    aggregated_df = aggregated_dfs[0]
    range_var_cols = [f"{col}_var" for col in range_cols]
    for df in aggregated_dfs[1:]: 
        aggregated_df[range_cols] += df[range_cols]
        aggregated_df[range_var_cols] += df[range_var_cols]
    # Compute the mean and variance across seeds
    aggregated_df[range_cols] /= len(seeds)
    aggregated_df[range_var_cols] /= len(seeds)
    aggregated_df[range_var_cols] -= aggregated_df[range_cols].rename(
        columns={range: var for range, var in zip(range_cols, range_var_cols)}
    ) ** 2
        

# --------------------- Define Plot Params ---------------------

metrics = [
    "range_res_norm",
    "range_spd_norm",
]
metrics_formatted = {
    "range_res_norm": "$\\hat{\\rho}^{\\text{res}}_{\\mathcal{G}$",
    "range_spd_norm": "$\\hat{\\rho}^{\\text{spd}}_{\\mathcal{G}$",
}

variance_metrics = [f"{metric}_var" for metric in metrics]
titles = [
    "Range Res Norm",
    "Range Spd Norm",
]

models = aggregated_df["model"].unique()
split = selected_splits[0]
scores = {}
for model, shortened in models_shortened.items():
    if model not in models:
        continue
    scores_over_seeds = [
        final_epoch_stats[model][seed][split][task_metric.lower()]
        for seed in seeds
    ]
    scores[model] = (
        shortened,
        np.mean(scores_over_seeds),
        np.std(scores_over_seeds),
    )

model_score_map = {
    model: f"{name:<7} {score:.3f}±{std:.2f}" for model, (name, score, std) in scores.items()
}

# Filter splits
if selected_splits:
    splits = [s for s in selected_splits if s in aggregated_df["split"].unique()]
    if not splits:
        print(
            "No matching splits found for the selected splits. Using all available splits."
        )
        splits = aggregated_df["split"].unique()
else:
    splits = aggregated_df["split"].unique()

color_map = plt.get_cmap("tab10")
model_colors = {model: color_map(i % 10) for i, model in enumerate(models)}

line_styles = {
    "train": "--",
    "val": "-",
}
for sp in splits:
    if sp not in line_styles:
        line_styles[sp] = "-"

# --------------------- Create the Plots ---------------------

# --- 1) User-defined width ratios to make right column narrower ---
column_width_ratios = [2, 1]  # Left column is twice as wide as the right column

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(8, 6),
    gridspec_kw={"width_ratios": column_width_ratios},
)
axes = axes.flatten()  # => [top-left, top-right, bottom-left, bottom-right]

# We'll map:
#   "range_res_norm" -> axes[0] (top-left)
#   "range_spd_norm" -> axes[2] (bottom-left)
metric_subplot_map = {
    "range_res_norm": axes[0],
    "range_spd_norm": axes[2],
}

# Sort models by descending score
sorted_models = sorted(models, key=lambda m: scores.get(m, ("", 0))[1], reverse={"MAE": False, "AP": True, "F1": True}[task_metric])

# --------------------- Plot Main Metrics ---------------------
for metric in metrics:
    ax = metric_subplot_map[metric]
    var_metric = f"{metric}_var"
    title = titles[metrics.index(metric)]

    for model in sorted_models:
        for sp in splits:
            df_plot = aggregated_df[
                (aggregated_df["model"] == model) & (aggregated_df["split"] == sp)
            ]
            if df_plot.empty:
                continue
            df_plot = df_plot.sort_values(by="epoch")

            std_dev = np.sqrt(df_plot[var_metric])
            # Only show label once per model (on the first split we plot)
            # so legend won't repeat it for train/val lines.
            label = model_score_map.get(model, model) if sp == splits[0] else ""
            ax.plot(
                df_plot["epoch"],
                df_plot[metric],
                marker="o",
                color=model_colors[model],
                linestyle=line_styles.get(sp, "-"),
                linewidth=2,
                label=label,
            )
            ax.fill_between(
                df_plot["epoch"],
                df_plot[metric] - std_dev,
                df_plot[metric] + std_dev,
                color=model_colors[model],
                alpha=0.2,
            )

    ax.set_ylabel(metrics_formatted[metric], fontsize=20, rotation=0)
    ax.grid(True)
    if logy:
        ax.set_yscale("log")
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    # Add x-label only for the bottom row
    if ax in [axes[2]]:
        ax.set_xlabel("Epoch", fontsize=14)

# --------------------- Add Zoomed-In Plot (ONLY for range_res_norm) ---------------------
zoom_metric = "range_res_norm"
zoom_var_metric = f"{zoom_metric}_var"
zoom_title = "Magnified"

ax_zoom = axes[1]  # top-right subplot

for model in sorted_models:
    for sp in splits:
        df_plot = aggregated_df[
            (aggregated_df["model"] == model) & (aggregated_df["split"] == sp)
        ]
        if df_plot.empty:
            continue
        df_plot = df_plot.sort_values(by="epoch")

        std_dev = np.sqrt(df_plot[zoom_var_metric])
        label = model_score_map.get(model, model) if sp == splits[0] else ""
        ax_zoom.plot(
            df_plot["epoch"],
            df_plot[zoom_metric],
            marker="o",
            color=model_colors[model],
            linestyle=line_styles.get(sp, "-"),
            linewidth=2,
            label=label,
        )
        ax_zoom.fill_between(
            df_plot["epoch"],
            df_plot[zoom_metric] - std_dev,
            df_plot[zoom_metric] + std_dev,
            color=model_colors[model],
            alpha=0.2,
        )

ax_zoom.set_title(zoom_title, fontsize=16)
ax_zoom.grid(True)
ax_zoom.set_xlim(-0.3, 5)
if logy:
    ax_zoom.set_yscale("log")
ax_zoom.tick_params(axis="x", labelsize=12)
ax_zoom.tick_params(axis="y", labelsize=12)

# --------------------- Custom Legend ---------------------
legend_elements = []

model_handles = [
    Line2D(
        [0],
        [0],
        color=model_colors[m],
        linestyle="-",
        linewidth=2,
        label=model_score_map.get(m, m),
    )
    for m in sorted_models
]
legend_elements.extend(model_handles)

if len(splits) > 1:
    split_colors = "gray"
    split_handles = []
    for sp in splits:
        linestyle = line_styles.get(sp, "-")
        split_handles.append(
            Line2D(
                [0],
                [0],
                color=split_colors,
                linestyle=linestyle,
                linewidth=2,
                label=sp.capitalize(),
            )
        )
    legend_elements.extend(split_handles)

error_bars_label = f"$\\sigma$ (over {len(seeds)} seeds)" if len(seeds) > 1 else f"$\\sigma$ (over {subset} graphs)"
std_dev_handle = Patch(facecolor="gray", edgecolor="w", alpha=0.2, label=error_bars_label)
legend_elements.append(std_dev_handle)

axes[3].axis("off")  # bottom-right subplot just for legend
legend = axes[3].legend(
    handles=legend_elements, loc="center", fontsize=10, prop=monospace_font
)
legend.set_title(f"Model: {task_metric}", prop=monospace_font)

# --------------------- Finalize and Save ---------------------
plt.tight_layout(rect=[0, 0, 0.85, 0.95])

filename = f"{plot_out_dir}/{dataset}_ranges{f'_subset{subset}' if subset > 0 else ''}_{'-'.join(sorted(models))}_{'-'.join(sorted(splits))}.png"
print(f"Saving plot to {filename}")
plt.savefig(filename, dpi=300, bbox_inches="tight")
plt.savefig(filename.replace(".png", ".pdf"), bbox_inches="tight")

plt.show()
