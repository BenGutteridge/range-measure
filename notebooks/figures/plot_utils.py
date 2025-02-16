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


def load_and_aggregate_data(path, dataset, seed=0, subset=500):
    """
    Loads and aggregates DataFrames from pickle files.
    """
    pattern = os.path.join(
        path, f"{dataset}-*", str(seed), "*", f"range_stats_subset{subset}_e*.pkl"
    )
    file_paths = glob.glob(pattern)

    if not file_paths:
        print("No data files found with the given pattern.")
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
            epoch = filename.split("_e")[-1].replace(".pkl", "")
            epoch = int(epoch)
        except (IndexError, ValueError):
            print(f"Skipping file with unexpected filename format: {file_path}")
            continue

        try:
            df = pd.read_pickle(file_path)
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
