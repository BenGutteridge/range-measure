"""
Notebook for plotting Cora range measure experiments.
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
from os.path import join
import pandas as pd
from loguru import logger
import json

monospace_font = font_manager.FontProperties(family="monospace", size=15)

# ------------------ Load and aggregate data ---------------------
plots_out_dir = "plots/rebuttals"
os.makedirs(plots_out_dir, exist_ok=True)
path = "range_results"
dataset = "amazon_ratings"  # cora, amazon_ratings, roman_empire
# Choose one range score to plot, e.g., 'range_res' or 'range_spd' or with "_norm"
range_key = "range_spd_norm"
split = "val"

sampling_kwargs = {
    "cora": {
        "in_node_prob": 1.0,
        "in_channel_prob": 0.2,
        "out_node_prob": 0.2,
        "out_channel_prob": 1.0,
    },
    "amazon_ratings": {
        "in_node_prob": 1.0,
        "in_channel_prob": 0.2,
        "out_node_prob": 0.2,
        "out_channel_prob": 1.0,
    },
    "roman_empire": {
        "in_node_prob": 1.0,
        "in_channel_prob": 0.2,
        "out_node_prob": 0.01,
        "out_channel_prob": 0.8,  # same as AR except this
    },
}[dataset]

models = [
    "GCN",
    "GT",
    "GCN+VN",
    # "GT_dense",
]
legend_replacements = {
    "GT_dense": "GT   ",  # fig legend formatting
}
folds = seeds = list(np.arange(10))

data = {model: {} for model in models}

for model in models:
    stats_path = join(
        path,
        dataset,
        model,
        split,
        f"sampling-nodes-in{sampling_kwargs['in_node_prob']:.2f}out{sampling_kwargs['out_node_prob']:.2f}-channels-in{sampling_kwargs['in_channel_prob']:.2f}-out{sampling_kwargs['out_channel_prob']:.2f}",
    )
    for fold in folds:
        try:
            fold_path = join(stats_path, str(fold))
            range_stats_str = "range_stats_+vn_f" if "+VN" in model else "range_stats_f"
            for file in os.listdir(fold_path):
                if file.startswith(range_stats_str) and file.endswith(".csv"):
                    stats = pd.read_csv(join(fold_path, file), index_col=0)
                    stats = stats.drop(columns=["var"])
                    fold = int(file.split("_")[-2].replace("fold", ""))
                    epoch = int(file.split("_e")[-1].split(".")[0])
                    data[model][epoch] = data[model].get(epoch, []) + [stats]
        except:
            logger.warning(f"Failed to load {model} fold {fold}")
for model in models:
    print(
        f"Model: {model} | epochs: {sorted(list(data[model].keys()))} | nums_per_epoch: {set([len(data[model][epoch]) for epoch in data[model]])}"
    )
    # Average over folds
    for epoch in data[model]:
        summed_stats, summed_stats_squared = 0, 0
        for fold_stats in data[model][epoch]:
            summed_stats += fold_stats
            summed_stats_squared += fold_stats**2
        data[model][epoch] = {
            "mean": summed_stats / len(data[model][epoch]),
            "std": np.sqrt(
                summed_stats_squared / len(data[model][epoch])
                - (summed_stats / len(data[model][epoch])) ** 2
            ),
        }
print(".")

# --------- Plotting with Style Matching -----------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Extract the rho type from range_key (last underscore-separated token)
rho_type = range_key.split("_")[1]

# Create some dummy scores for the legend; key = model name, val = score
# (We're using dummy numbers; update if real scores are desired.)

scores = {}
for model in models:
    try:
        with open(join("range_results", dataset, model + ".json"), "r") as f:
            scores[model] = json.load(f)[split]
    except:
        scores[model] = 0.453  # .71 for GT dense on cora approx


# Sort models by descending score for the legend
sorted_models = sorted(
    models,
    key=lambda m: scores.get(m, 0),  # default to 0 if model not in scores
    reverse=True,
)

# Build label strings like "GCN   0.123" for the legend
model_score_map = {m: f"{m:<5} {scores.get(m,0):.3f}" for m in models}

# Extract epochs (assumes same for all models)
epochs = sorted(next(iter(data.values())).keys())

# Figure setup: single subplot, sized to match the example
fig, ax = plt.subplots(figsize=(6, 3))

# Colors for lines
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

# Plot lines with markers + shaded std
for i, model in enumerate(sorted_models):
    means = [data[model][epoch]["mean"].loc[range_key].item() for epoch in epochs]
    stds = [data[model][epoch]["std"].loc[range_key].item() for epoch in epochs]
    means = np.array(means)
    stds = np.array(stds)

    ax.plot(
        epochs,
        means,
        color=colors[i % len(colors)],
        marker="o",
        markersize=6,
        linewidth=2,
        label=(
            model_score_map[model] if i == 0 else ""
        ),  # We'll build a custom legend below
    )
    ax.fill_between(
        epochs,
        means - stds,
        means + stds,
        color=colors[i % len(colors)],
        alpha=0.2,
        linewidth=0,
    )

# y-axis log scale
ax.set_yscale("log")
# ax.set_xscale("log")

# Axis labels
ax.set_xlabel("Epoch", fontproperties=monospace_font)
# ax.set_ylabel(rf"$\\rho_{{\\mathrm{{{rho_type}}}}}$", fontproperties=monospace_font)
# rotate 90 degrees
ax.set_ylabel(f"$\\rho_{{{rho_type}}}$", fontproperties=monospace_font, rotation=0)

# ax.set_title(f"{dataset.capitalize()} {split.capitalize()}")

# Grid styling
ax.grid(True, which="both", linestyle="-", alpha=0.5)

# Create custom legend elements for each model in sorted order
legend_elements = []
for i, m in enumerate(sorted_models):
    label = model_score_map[m]
    for orig, repl in legend_replacements.items():
        label = label.replace(orig, repl)
    line = Line2D([0], [0], color=colors[i % len(colors)], linewidth=2, label=label)
    legend_elements.append(line)

# Create a patch for the standard deviation shading
std_dev_patch = Patch(facecolor="gray", edgecolor="w", alpha=0.2, label="$\\sigma$")
legend_elements.append(std_dev_patch)

# Place the legend outside the plot to the right
# "Model: Score" as the legend title
legend = fig.legend(
    handles=legend_elements,
    loc="center left",
    bbox_to_anchor=(0.87, 0.5),
    borderaxespad=0.1,
    prop=monospace_font,
)
legend.set_title("Model: Acc", prop=monospace_font)

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 0.85, 1])

# Save figure
plot_path = f"{plots_out_dir}/{dataset}_{range_key}_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.savefig(plot_path.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Plot saved to: {plot_path}")

# Display
plt.show()
