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

path = "lrgb_exps/results/sota"
dataset = "vocsuperpixels"
subset = 500
seed = 0
logy = True  # Set log-scale y axis

selected_splits = [
    "val",
]

models_shortened = {
    "GatedGCN": "GGCN",
    "GCN": "GCN",
    "GPS": "GPS",
    "GINE": "GINE",
}

final_epoch_stats = {
    model: get_final_epoch_stats(os.path.join(path, f"{dataset}-{model}"))
    for model in models_shortened.keys()
}

task_metric = {
    "vocsuperpixels": "F1",
    "peptides-func": "AP",
    "peptides-struct": "MAE",
}[dataset]


aggregated_df = load_and_aggregate_data(
    path=path,
    dataset=dataset,
    subset=subset,
    seed=seed,
)

# --------------------- Define Plot Params ---------------------

metrics = [
    "range_res_norm",
    "range_spd_norm",
]
metrics_formatted = {
    "range_res_norm": "$\\rho_{\\text{res}}$",
    "range_spd_norm": "$\\rho_{\\text{spd}}$",
}

variance_metrics = [f"{metric}_var" for metric in metrics]
titles = [
    "Range Res Norm",
    "Range Spd Norm",
]

models = aggregated_df["model"].unique()


split = selected_splits[0]
scores = {
    model: (
        shortened,
        final_epoch_stats[model][seed][split][task_metric.lower()],
    )
    for model, shortened in models_shortened.items()
}
model_score_map = {m: f"{name:<5} {val:.3f}" for m, (name, val) in scores.items()}

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
sorted_models = sorted(models, key=lambda m: scores.get(m, ("", 0))[1], reverse=True)

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

# --------------------- REMOVE or COMMENT OUT THE PROBLEMATIC LINE ---------------------
# The following line was disabling all tick labels on whichever axis was last used by 'ax':
# ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

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

std_dev_handle = Patch(facecolor="gray", edgecolor="w", alpha=0.2, label="$\sigma$")
legend_elements.append(std_dev_handle)

axes[3].axis("off")  # bottom-right subplot just for legend
legend = axes[3].legend(
    handles=legend_elements, loc="center", fontsize=10, prop=monospace_font
)
legend.set_title(f"Model: {task_metric}", prop=monospace_font)

# --------------------- Finalize and Save ---------------------
plt.tight_layout(rect=[0, 0, 0.85, 0.95])

filename = f"plots/{dataset}_ranges{f'_subset{subset}' if subset > 0 else ''}_{'-'.join(sorted(models))}_{'-'.join(sorted(splits))}.png"
print(f"Saving plot to {filename}")
plt.savefig(filename, dpi=300, bbox_inches="tight")
plt.savefig(filename.replace(".png", ".pdf"), bbox_inches="tight")

plt.show()
