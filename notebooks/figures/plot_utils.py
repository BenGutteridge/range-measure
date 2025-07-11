import pandas as pd
import os
from os import path as osp
import json
from loguru import logger
import glob


def get_final_epoch_stats(run_dir: str) -> dict:
    """Retrieve the final epoch stats for all seeds and splits in the LRGB GraphGPS run directory."""
    splits = ["train", "val", "test"]
    res = {}

    for seed in os.listdir(run_dir):
        # check if integer
        if not seed.isdigit():
            continue
        seed = int(seed)
        res[seed] = {}
        for split in splits:
            filepath = osp.join(run_dir, str(seed), split, "stats.json")
            if osp.exists(filepath):
                # Load jsonl file last line
                with open(filepath, "r") as f:
                    lines = f.readlines()
                    final_epoch_stats = json.loads(lines[-1])
                res[seed][split] = final_epoch_stats
                logger.info(
                    f"Loaded final epoch stats for seed {seed} split {split} at {run_dir.split('/')[-1]}"
                )
    return res


def load_and_aggregate_data(path, dataset, seed=0, subset=500, **sampling_kwargs):
    """
    Loads and aggregates DataFrames from pickle files.
    """
    if sampling_kwargs:
        ch_in, ch_out, nod_in, nod_out, max_nod, min_nod, samp_seed = (
            sampling_kwargs["in_channel_prob"],
            sampling_kwargs["out_channel_prob"],
            sampling_kwargs["in_node_prob"],
            sampling_kwargs["out_node_prob"],
            sampling_kwargs["max_nodes"],
            sampling_kwargs["min_nodes"],
            sampling_kwargs.get("sampling_seed", seed),
        )
        pattern = os.path.join(
            path,
            f"{dataset}-*",
            str(seed),
            "*",
            f"range_stats_subset{subset}_samples-channels-in{ch_in:.2f}out{ch_out:.2f}-nodes-in{nod_in:.2f}out{nod_out:.2f}-min{min_nod}-max{max_nod}_seed{samp_seed}_e*.csv",
        )
    else:
        pattern = os.path.join(
            path, f"{dataset}-*", str(seed), "*", f"range_stats_subset{subset}_e*.csv"
        )
    file_paths = glob.glob(pattern)

    if not file_paths:
        logger.warning("No data files found with the given pattern.")
        return pd.DataFrame()

    data_list = []
    for file_path in sorted(file_paths):
        parts = file_path.split(os.sep)
        if len(parts) < 5:
            print(f"Skipping file with unexpected path format: {file_path}")
            continue

        dataset_model = parts[-4]  # 'dataset-model'
        split = parts[-2]  # 'val', 'train', etc.
        filename = parts[-1]  # 'range_stats_subset500_e000.pkl'

        try:
            model = dataset_model.split("-")[-1]
        except ValueError:
            print(f"Skipping file with unexpected dataset-model format: {file_path}")
            continue

        try:
            epoch = int(filename.split("_e")[-1].replace(".csv", ""))

        except (IndexError, ValueError):
            print(f"Skipping file with unexpected filename format: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path, index_col=0)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        df = df.reset_index().rename(columns={"index": "metric"})

        data = {"model": model, "split": split, "epoch": epoch}
        for _, row in df.iterrows():
            metric = row["metric"]
            value = row["range"]
            variance = row["var"]
            data[metric] = value
            data[f"{metric}_var"] = variance

        data_list.append(data)

    aggregated_df = pd.DataFrame(data_list)
    if aggregated_df.empty:
        print("Aggregated DataFrame is empty after processing.")
        return aggregated_df

    aggregated_df.sort_values(by=["model", "split", "epoch"], inplace=True)
    aggregated_df.reset_index(drop=True, inplace=True)

    return aggregated_df
