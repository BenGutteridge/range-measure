import logging
import pickle
import torch
from torch_geometric.utils import subgraph
from tqdm import tqdm
import os
from os.path import join
from loguru import logger
import time


def pre_transform_in_memory(dataset, transform_func, show_progress=False):
    """Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return dataset

    # Check for positional encoding stats
    args = transform_func.keywords
    pe_strs = args.get("pe_types", [])
    posenc_filepath = join(
        args["cfg"].dataset.dir,
        f"{args['cfg'].dataset.format}-{args['cfg'].dataset.name}_{'-'.join(pe_strs)}_{transform_func.func.__name__}.pkl",
    )  # if not a positional encoding transform, will be an unused dummy filename
    if (
        args["cfg"].dataset.format in ["PyG-VOCSuperpixels", "PyG-ZINC"]
        and args["cfg"].longrange.subset > 0
    ):
        posenc_filepath = posenc_filepath.replace(
            ".pkl", f"_subset{args['cfg'].longrange.subset}.pkl"
        )  # VOC subset changes the Dataset object directly, so we need a separate pos enc for each subset

    # Load saved positional encoding stats and add to dataset
    if os.path.exists(posenc_filepath):
        logging.info(f"Loading positional encoding stats from {posenc_filepath}...")
        with open(posenc_filepath, "rb") as f:
            pos_enc = pickle.load(f)

        if pe_strs == ["RWSE"]:
            pos_enc_rwse = pos_enc["RWSE"]

            def add_rwse(data, rwse):
                data.pestat_RWSE = rwse
                return data

            t0 = time.time()
            data_list = [
                add_rwse(dataset.get(i), pos_enc_rwse[i])
                for i in tqdm(
                    range(len(dataset)),
                    disable=not show_progress,
                    mininterval=10,
                    miniters=len(dataset) // 20,
                )
            ]
        elif pe_strs == ["LapPE"]:

            def add_lappe(data, eigvals, eigvecs):
                data.EigVals = eigvals
                data.EigVecs = eigvecs
                return data

            t0 = time.time()
            data_list = [
                add_lappe(dataset.get(i), pos_enc["EigVals"][i], pos_enc["EigVecs"][i])
                for i in tqdm(
                    range(len(dataset)),
                    disable=not show_progress,
                    mininterval=10,
                    miniters=len(dataset) // 20,
                )
            ]
        logging.info("...done.")
        logging.info(
            f"Appended {len(data_list)} saved examples in {(time.time() - t0):.2f}s"
        )

    else:
        t0 = time.time()
        data_list = [
            transform_func(dataset.get(i))
            for i in tqdm(
                range(len(dataset)),
                disable=not show_progress,
                mininterval=10,
                miniters=len(dataset) // 20,
            )
        ]
        logger.info(
            f"Performed {transform_func.func.__name__} transform for {len(data_list)} examples in {(time.time() - t0):.2f}s"
        )
        # Extract and save pos_enc stats
        pos_enc = {}
        for pe_str in pe_strs:
            logging.info(
                f"Saving {pe_str} positional encoding stats to {posenc_filepath}..."
            )
            for d in data_list:
                if pe_str == "RWSE":
                    pos_enc["RWSE"] = pos_enc.get("RWSE", []) + [d.pestat_RWSE]
                elif pe_str == "LapPE":
                    pos_enc["EigVals"] = pos_enc.get("EigVals", []) + [d.EigVals]
                    pos_enc["EigVecs"] = pos_enc.get("EigVecs", []) + [d.EigVecs]
                else:
                    raise NotImplementedError(f"{pe_str} not supported.")
        if pos_enc:
            with open(posenc_filepath, "wb") as f:
                pickle.dump(pos_enc, f)
            logging.info("...done.")
    data_list = list(filter(None, data_list))
    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)


def typecast_x(data, type_str):
    if type_str == "float":
        data.x = data.x.float()
    elif type_str == "long":
        data.x = data.x.long()
    else:
        raise ValueError(f"Unexpected type '{type_str}'.")
    return data


def concat_x_and_pos(data):
    data.x = torch.cat((data.x, data.pos), 1)
    return data


def clip_graphs_to_size(data, size_limit=5000):
    if hasattr(data, "num_nodes"):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    if N <= size_limit:
        return data
    else:
        logging.info(f"  ...clip to {size_limit} a graph of size: {N}")
        if hasattr(data, "edge_attr"):
            edge_attr = data.edge_attr
        else:
            edge_attr = None
        edge_index, edge_attr = subgraph(
            list(range(size_limit)), data.edge_index, edge_attr
        )
        if hasattr(data, "x"):
            data.x = data.x[:size_limit]
            data.num_nodes = size_limit
        else:
            data.num_nodes = size_limit
        if hasattr(data, "node_is_attributed"):  # for ogbg-code2 dataset
            data.node_is_attributed = data.node_is_attributed[:size_limit]
            data.node_dfs_order = data.node_dfs_order[:size_limit]
            data.node_depth = data.node_depth[:size_limit]
        data.edge_index = edge_index
        if hasattr(data, "edge_attr"):
            data.edge_attr = edge_attr
        return data
