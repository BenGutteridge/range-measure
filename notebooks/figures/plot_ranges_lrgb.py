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
import sys

# --------------------- Global Font Sizes ---------------------
base_fontsize = 10       # Font size for everything
ylabel_fontsize = 15     # Font size for y-axis label only

# Define a monospace font for legend title if desired
monospace_font_title = font_manager.FontProperties(family="monospace", size=base_fontsize)
# Use a smaller size for the legend items text
legend_prop = {"family": "monospace", "size": base_fontsize}

# ------------------ Load and aggregate data ---------------------

plot_out_dir = "plots/camera_ready"
os.makedirs(plot_out_dir, exist_ok=True)
path = "lrgb_exps/results/camera_ready"
if "func" in sys.argv[1:]:
    dataset = "peptides-func"
elif "struct" in sys.argv[1:]:
    dataset = "peptides-struct"
else:
    dataset = "vocsuperpixels"
metric = "range_res_norm" if "res" in sys.argv[1:] else "range_spd_norm"
subset = 500 if dataset == "vocsuperpixels" else 200
seeds = [0, 1, 2, 3]
logy = False if dataset=="vocsuperpixels" and metric=="range_res_norm" else True  # Set log-scale y axis
zoomed_lims = [1, 5]  # epochs for zoomed (early epochs)
zoomed_xticks = [1, 1, 2, 3, 4]
normal_lims = [5, 101]  # epochs for full range
normal_xticks = [5, 20, 40, 60, 80, 100]

sampling_kwargs = {
    "in_node_prob": 1.0,
    "in_channel_prob": 0.5,
    "out_node_prob": 0.5,
    "out_channel_prob": 0.5,
    "min_nodes": 16,
    "max_nodes": 256,
}
selected_splits = ["val"]
subset = {
    "vocsuperpixels": 500,
    "peptides-func": 200,
    "peptides-struct": 200,
}[dataset]

models_shortened = {
    "GatedGCN": "GatedGCN",
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
metric_higher_is_better = {
    "MAE": False, 
    "AP": True, 
    "F1": True,
}[task_metric]


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

aggregated_df = aggregated_dfs[0]
if len(seeds) > 1:
    range_cols = [col for col in aggregated_dfs[0].columns if col.startswith("range") and not col.endswith("_var")]
    for i in range(len(aggregated_dfs)):
        for col in range_cols:
            aggregated_dfs[i][f"{col}_var"] = aggregated_dfs[i][col] ** 2
    aggregated_df = aggregated_dfs[0]
    range_var_cols = [f"{col}_var" for col in range_cols]
    for df in aggregated_dfs[1:]:
        aggregated_df[range_cols] += df[range_cols]
        aggregated_df[range_var_cols] += df[range_var_cols]
    aggregated_df[range_cols] /= len(seeds)
    aggregated_df[range_var_cols] /= len(seeds)
    aggregated_df[range_var_cols] -= aggregated_df[range_cols].rename(
        columns={range: var for range, var in zip(range_cols, range_var_cols)}
    ) ** 2

# Shift epochs to start from 0
aggregated_df["epoch"] = aggregated_df["epoch"] - aggregated_df["epoch"].min()

# --------------------- Define Plot Params ---------------------

metrics_formatted = {
    "range_res_norm": "$\\hat{\\rho}^{\\text{res}}_\\mathcal{G}$",
    "range_spd_norm": "$\\hat{\\rho}^{\\text{spd}}_\\mathcal{G}$",
}

models = sorted(aggregated_df["model"].unique())
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
    model: f"{name:<8} {score:.3f}±{std:.2f}".replace("0.", ".")
    for model, (name, score, std) in scores.items()
}

if selected_splits:
    splits = [s for s in selected_splits if s in aggregated_df["split"].unique()]
    if not splits:
        print("No matching splits found for the selected splits. Using all available splits.")
        splits = aggregated_df["split"].unique()
else:
    splits = aggregated_df["split"].unique()

color_map = plt.get_cmap("tab10")
model_colors = {model: color_map(i % 10) for i, model in enumerate(models)}

# --------------------- Assign a unique marker per model ---------------------
marker_list = ['o', 's', '^', 'D', 'v', 'd', '*', 'P', 'X', 'h']
model_markers = {
    model: marker_list[i % len(marker_list)]
    for i, model in enumerate(models)
}

line_styles = {
    "train": "--",
    "val": "-",
}
for sp in splits:
    if sp not in line_styles:
        line_styles[sp] = "-"

# --------------------- Create the Plots ---------------------
fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(4.5, 2),
    gridspec_kw={"width_ratios": [1.2, 1.8, 1.52], "wspace": 0.15}
)
ax_magnified = axes[0]
ax_normal = axes[1]
ax_legend = axes[2]

var_metric = f"{metric}_var"

# ------------------ Left Plot: Zoomed (Magnified) View ------------------
# ax_magnified.set_title(titles["Magnified"], fontsize=base_fontsize)
# Set the entire background of the zoomed plot to a light grey color.
import matplotlib.colors as mcolors
# ax_magnified.set_facecolor(mcolors.to_rgba('lightgrey', alpha=0.3))

for model in sorted(models, key=lambda m: scores.get(m, ("", 0))[1],
                    reverse=metric_higher_is_better):
    for sp in splits:
        df_plot = aggregated_df[(aggregated_df["model"] == model) & (aggregated_df["split"] == sp)]
        if df_plot.empty:
            continue
        df_plot = df_plot.sort_values(by="epoch")
        std_dev = np.sqrt(df_plot[var_metric])
        label = model_score_map.get(model, model) if sp == splits[0] else ""
        ax_magnified.plot(
            df_plot["epoch"],
            df_plot[metric],
            marker=model_markers[model],
            markersize=5,
            color=model_colors[model],
            linestyle=line_styles.get(sp, "-"),
            linewidth=2,
            label=label,
        )
        ax_magnified.fill_between(
            df_plot["epoch"],
            df_plot[metric] - std_dev,
            df_plot[metric] + std_dev,
            color=model_colors[model],
            alpha=0.2,
        )

ax_magnified.set_ylabel(metrics_formatted[metric], fontsize=ylabel_fontsize, rotation=0)
ax_magnified.set_xlabel("Epoch", fontsize=base_fontsize)
ax_magnified.set_xlim(zoomed_lims[0], zoomed_lims[1])
ax_magnified.set_xticks(zoomed_xticks)
ax_magnified.grid(True)
if logy:
    ax_magnified.set_yscale("log")
ax_magnified.tick_params(axis="x", labelsize=base_fontsize)
ax_magnified.tick_params(axis="y", labelsize=base_fontsize)

ax_ylims = ax_magnified.get_ylim() # use for setting the size for the other plot

# ------------------ Middle Plot: Remaining Epochs View ------------------
for model in sorted(models, key=lambda m: scores.get(m, ("", 0))[1],
                    reverse=metric_higher_is_better):
    for sp in splits:
        df_plot = aggregated_df[(aggregated_df["model"] == model) & (aggregated_df["split"] == sp)]
        if df_plot.empty:
            continue
        # Cut off only included epochs (except 0, to keep the overall scale consistent)
        df_plot = df_plot[(df_plot["epoch"] >= normal_lims[0]) & (df_plot["epoch"] <= normal_lims[1])]
        df_plot = df_plot.sort_values(by="epoch")
        std_dev = np.sqrt(df_plot[var_metric])
        label = model_score_map.get(model, model) if sp == splits[0] else ""
        ax_normal.plot(
            df_plot["epoch"],
            df_plot[metric],
            marker=model_markers[model],
            markersize=5,
            color=model_colors[model],
            linestyle=line_styles.get(sp, "-"),
            linewidth=2,
            label=label,
        )
        ax_normal.fill_between(
            df_plot["epoch"],
            df_plot[metric] - std_dev,
            df_plot[metric] + std_dev,
            color=model_colors[model],
            alpha=0.2,
        )

ax_normal.set_xlabel("Epoch", fontsize=base_fontsize)
ax_normal.set_xlim(normal_lims[0], normal_lims[1])
ax_normal.set_xticks(normal_xticks)
# Instead of a vertical dashed line, shade the background region from the beginning up to zoomed_lims[1] as light grey.
ax_normal.axvspan(normal_lims[0], zoomed_lims[1], facecolor='lightgrey', alpha=0.3)
ax_normal.axvline(x=zoomed_lims[1], color='black', linestyle='--', linewidth=0.6)
ax_normal.set_ylim(ax_ylims[0], ax_ylims[1]) # Set the y-limits to be the same as the zoomed plot
ax_normal.grid(True)
if logy:
    ax_normal.set_yscale("log")
ax_normal.tick_params(axis="x", labelsize=base_fontsize)
ax_normal.tick_params(axis="y", labelsize=base_fontsize, labelleft=False)

# ------------------ Custom Legend ------------------
legend_elements = []
# Create legend handles for the models with both line and marker
model_handles = [
    Line2D(
        [0],
        [0],
        color=model_colors[m],
        marker=model_markers[m],
        markersize=5,
        linestyle="-",
        linewidth=2,
        label=model_score_map.get(m, m),
    )
    for m in sorted(models, key=lambda m: scores.get(m, ("", 0))[1],
                    reverse=metric_higher_is_better)
]
legend_elements.extend(model_handles)

if len(splits) > 1:
    split_handles = []
    for sp in splits:
        split_handles.append(
            Line2D(
                [0],
                [0],
                color="gray",
                linestyle=line_styles.get(sp, "-"),
                linewidth=2,
                label=sp.capitalize(),
            )
        )
    legend_elements.extend(split_handles)

error_bars_label = f"$\\sigma$ ({len(seeds)} seeds)" if len(seeds) > 1 else f"$\\sigma$ ({subset} graphs)"
std_dev_handle = Patch(facecolor="gray", edgecolor="w", alpha=0.2, label=error_bars_label)
legend_elements.append(std_dev_handle)

ax_legend.axis("off")
ax_legend.legend(
    handles=legend_elements,
    loc="center",
    bbox_to_anchor=(0.82, 0.5),
    prop=legend_prop,
    title=f"Model: {task_metric}{'↑' if metric_higher_is_better else '↓'}",
)

plt.tight_layout(rect=[0, 0, 0.95, 0.95])
filename = f"{plot_out_dir}/{dataset}_{metric}{f'_subset{subset}' if subset > 0 else ''}_{'-'.join(sorted(models))}_{'-'.join(sorted(splits))}.png"
print(f"Saving plot to {filename}")
plt.savefig(filename, dpi=300, bbox_inches="tight")
plt.savefig(filename.replace(".png", ".pdf"), bbox_inches="tight")
plt.show()
