"""
Notebook for plotting LRGB range measure experiments for LRGB peptides-func and -struct.

Generates Figures 7, 8, 9 in the paper.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from math import ceil
from matplotlib import font_manager
from notebooks.figures.plot_utils import (
    get_final_epoch_stats,
    load_and_aggregate_data,
)

monospace_font = font_manager.FontProperties(family="monospace", size=15)

# ------------------ Load and aggregate data ---------------------

path = "lrgb_exps/results/sota"
dataset = "peptides-func" # **** change to peptides-struct for struct plot *****
subset = 200
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

# Define the metrics and their variance columns
metrics = [
    "range_res_norm",
    # "range_spd_norm",
    # "range_res",
    # "range_spd",
]  # formatting params set up to plot one metric at a time

metrics_formatted = {
    "range_res_norm": "$\\rho_{\\text{res}}$",
    "range_spd_norm": "$\\rho_{\\text{spd}}$",
}

variance_metrics = [f"{metric}_var" for metric in metrics]
titles = [
    "Range Res Norm",
    "Range Spd Norm",
    # "Range Res",
    # "Range Spd",
]

# Get unique models and splits
models = aggregated_df["model"].unique()

split = selected_splits[0]  # Use first split in list for scores

scores = {
    model: (
        shortened,
        final_epoch_stats[model][seed][split][task_metric.lower()],
    )
    for model, shortened in models_shortened.items()
}

model_score_map = {
    model: f"{name:<5} {score:.3f}" for model, (name, score) in scores.items()
}

# Filter splits based on user selection
if selected_splits:
    splits = [s for s in selected_splits if s in aggregated_df["split"].unique()]
    if not splits:
        print(
            "No matching splits found for the selected splits. Using all available splits."
        )
        splits = aggregated_df["split"].unique()
else:
    splits = aggregated_df["split"].unique()

# Assign a unique color to each model using a colormap
color_map = plt.get_cmap("tab10")  # 'tab10' has 10 distinct colors
model_colors = {
    model: color_map(i % 10) for i, model in enumerate(models)
}  # Handles more than 10 models

# Define line styles for each split
line_styles = {
    "train": "--",  # Dashed line for train
    "val": "-",  # Solid line for val
    # Add more splits and styles if necessary
}

# Ensure that all splits have a defined line style
for split in splits:
    if split not in line_styles:
        line_styles[split] = "-"  # Default to solid line if not defined

# --------------------- Create the Plots ---------------------

# Define the grid size (maintaining vertical structure similar to original)
num_metrics = len(metrics)
nrows = 2
ncols = ceil(num_metrics / nrows)
fig, axes = plt.subplots(
    nrows, ncols, figsize=(6, 3.5 * nrows)
)  # Adjust figsize as needed
axes = axes.flatten()  # Flatten the axes array for easy iteration

# Sort models by score in descending order
sorted_models = sorted(models, key=lambda m: scores.get(m, ("", 0))[1], reverse=True)

# Iterate over each subplot and corresponding metric
for i, ax in enumerate(axes[:num_metrics]):
    metric = metrics[i]
    var_metric = variance_metrics[i]
    title = titles[i]

    for model in sorted_models:
        for split in splits:
            # Filter the DataFrame for the current model and split
            df_plot = aggregated_df[
                (aggregated_df["model"] == model) & (aggregated_df["split"] == split)
            ]

            if df_plot.empty:
                continue  # Skip if no data for this combination

            # Sort by epoch to ensure proper plotting
            df_plot = df_plot.sort_values(by="epoch")

            # Compute standard deviation from variance
            std_dev = np.sqrt(df_plot[var_metric])

            # Plot the main line for the current model and split
            label = (
                model_score_map.get(model, model)
                if (split == splits[0] and i == 0)
                else ""
            )
            ax.plot(
                df_plot["epoch"],
                df_plot[metric],
                marker="o",
                color=model_colors[model],
                linestyle=line_styles.get(split, "-"),
                linewidth=2,
                label=label,
            )

            # Plot the standard deviation as shaded error bars
            ax.fill_between(
                df_plot["epoch"],
                df_plot[metric] - std_dev,
                df_plot[metric] + std_dev,
                color=model_colors[model],
                alpha=0.2,  # Transparent shading for error bars
            )

    # Add titles and labels
    # ax.set_title(title, fontsize=14)
    # if bottom of column, add x label axis
    if i >= num_metrics - ncols:
        ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel(metrics_formatted[metric], fontsize=24, rotation=0)
    ax.grid(True)
    if logy:
        ax.set_yscale("log")
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)


# Remove any unused subplots
for j in range(num_metrics, len(axes)):
    fig.delaxes(axes[j])

# --------------------- Custom Legend Creation ---------------------

legend_elements = []

# Create custom legend handles for models sorted by score (highest to lowest)
model_handles = [
    Line2D(
        [0],
        [0],
        color=model_colors[model],
        linestyle="-",
        linewidth=2,
        label=model_score_map.get(model, model),
    )
    for model in sorted_models
]
legend_elements.extend(model_handles)

# Include split handles only if more than one split is selected
if len(splits) > 1:
    # Create custom legend handles for splits
    split_colors = "gray"  # All splits use grey color
    split_handles = []
    for split in splits:
        linestyle = line_styles.get(split, "-")
        split_handles.append(
            Line2D(
                [0],
                [0],
                color=split_colors,
                linestyle=linestyle,
                linewidth=2,
                label=split.capitalize(),  # Capitalize for better presentation
            )
        )
    legend_elements.extend(split_handles)

# Create a legend handle for the standard deviation
std_dev_handle = Patch(facecolor="gray", edgecolor="w", alpha=0.2, label="$\\sigma$")
legend_elements.append(std_dev_handle)

# --------------------- Finalize and Save Plot ---------------------

# # Add an overall title to the figure
# fig.suptitle(
#     f"{dataset}\nSubset: {subset} | Seed {seed}",
#     fontsize=16,
#     weight="bold",
# )

# Add the legend outside the plotting area, closer to the plots
bbox_to_anchor = (0.87, 0.5) if len(metrics) == 2 else (0.87, 0.7)
legend = fig.legend(
    handles=legend_elements,
    loc="center left",
    fontsize=18,
    bbox_to_anchor=bbox_to_anchor,  # Adjusted to be closer to the plots
    borderaxespad=0.1,
    prop=monospace_font,
)

# Add a legend title 'Model: Score'
legend.set_title(f"Model: {task_metric}", prop=monospace_font)

# Adjust layout for better visualization
plt.tight_layout(
    rect=[0, 0, 0.85, 0.95]
)  # Adjust rect to make room for the legend and suptitle

# Save the figure
filename = f"plots/{dataset}_ranges{f'_subset{subset}' if subset > 0 else ''}_{'-'.join(sorted(models))}_{'-'.join(sorted(splits))}_{'-'.join(sorted(metrics))}.png"
print(f"Saved to {filename}")
plt.savefig(
    filename, dpi=300, bbox_inches="tight"
)  # Use bbox_inches='tight' to include the legend
# Save as pdf as well
plt.savefig(filename.replace(".png", ".pdf"), bbox_inches="tight")

# Display the plot
plt.show()
