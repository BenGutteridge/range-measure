import torch

from torch_geometric.graphgym.checkpoint import load_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_train

from longrange.range import compute_range, DiffModel, HeadlessGraphLevelModelWrapper
from loguru import logger as log
from tqdm import tqdm
import os
from os import path as osp
import sys
import pandas as pd
from loguru import logger


@register_train("eval_range")
def eval_range(
    _,
    loaders,
    model,
    optimizer,
    scheduler,
    use_hessian=False,
):
    """
    Customized training pipeline.

    Args:
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler
        use_hessian: Whether to compute final result Hessian instead of pre-pooling Jacobian for graph-level tasks.

    """
    # Run params
    split = cfg.longrange.split
    subset = cfg.longrange.subset
    epoch = cfg.longrange.epoch
    is_graph_level_task = "peptides" in cfg.dataset.name or "ZINC" in cfg.dataset.format
    use_prepooling_layer = (
        is_graph_level_task and not use_hessian
    )  # use final node-level layer before pooling as output for graph-level tasks
    loader = {"train": loaders[0], "val": loaders[1], "test": loaders[2]}[split]
    input_dim = next(iter(loader)).x.shape[1]
    output_dim = cfg.gnn.dim_inner if use_prepooling_layer else cfg.share.dim_out
    # Number of channels to sample, if sampling
    num_out_channel = round(output_dim * cfg.longrange.out_channel_prob)
    num_in_channel = round(input_dim * cfg.longrange.in_channel_prob)
    clamp_nodes = lambda n: (
        min(
            cfg.longrange.max_nodes or float("inf"),  # default is 0 -> inf
            max(n, cfg.longrange.min_nodes),
        )
    )  # for node sampling within bounds (currently used for output nodes only)

    # Load model at specified epoch
    if epoch == -1: # hacky, perhaps in future we should set epochs to be +1 current
        ckpt_path = os.path.join(cfg.run_dir, "ckpt", f"{epoch}.ckpt")
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        logger.info(f"Loaded model at initialization.")
    else:
        loaded_epoch = (
            load_ckpt(model, optimizer, scheduler, epoch) - 1
        )  # load_ckpt returns epoch+1 if successful, 0 otherwise
        logger.info(f"Loaded model at epoch {loaded_epoch}.")
        if loaded_epoch == -1:
            raise ValueError(
                f"Failed to load model for epoch {epoch}, ckpt file likely not found."
            )
        if loaded_epoch != epoch:
            logger.warning(
                f"\nFailed to load model for epoch {epoch}, loaded epoch {loaded_epoch} instead."
            )
            epoch = loaded_epoch

    # Define paths and filenames for logging
    file_id = f"{f'subset{subset}' if subset > 0 else ''}"
    # Display sampling parameters
    logger.info(
        f"Using sampled Jacobian with:\n{cfg.longrange.in_channel_prob} input channel probability"
        f"\n{cfg.longrange.out_channel_prob} output channel probability\n{cfg.longrange.in_node_prob} input node probability"
        f"\n{cfg.longrange.out_node_prob} output node probability"
        f"\nMin {cfg.longrange.min_nodes} nodes\nMax {cfg.longrange.max_nodes or float('inf')} nodes"
        f"\nInput dim: {input_dim}\nOutput dim: {output_dim}"
        f"\nSampling seed: {cfg.longrange.sampling_seed}"
    )
    # Adjust file_id accordingly
    file_id += f"_samples-channels-in{cfg.longrange.in_channel_prob:.2f}out{cfg.longrange.out_channel_prob:.2f}-nodes-in{cfg.longrange.in_node_prob:.2f}out{cfg.longrange.out_node_prob:.2f}"
    if cfg.longrange.min_nodes > 0:
        file_id += f"-min{cfg.longrange.min_nodes}"
    if cfg.longrange.max_nodes > 0:
        file_id += f"-max{cfg.longrange.max_nodes}"
    file_id += f"_seed{cfg.longrange.sampling_seed}"
    file_id += f"_e{epoch:03d}"

    ranges_filename = f"{cfg.run_dir}/{split}/range_stats_{file_id}.csv"
    os.makedirs(f"{cfg.run_dir}/{split}/jacobians", exist_ok=True)
    gpu_stats_filename = f"{cfg.run_dir}/{split}/gpu_stats_{file_id}.csv"  # for tracking GPU usage stats; time, peak memory, etc.
    jacs_filename = f"{cfg.run_dir}/{split}/jacobians/jacobians_{file_id}.pt"

    # Check if range measures already computed
    if osp.exists(ranges_filename):
        log.info(f"Range stats already computed for split {split} and epoch {epoch}.")
        sys.exit(0)

    # If not, compute range measures
    log.info(f"Calculating range measure.\nSplit: {split}" f"\nEpoch: {epoch}")
    model.eval()
    range_stats, range_vars = (
        {
            "range_res": 0.0,
            "range_res_norm": 0.0,
            "range_spd": 0.0,
            "range_spd_norm": 0.0,
        }
        for _ in range(2)
    )

    try:
        jacobians = torch.load(jacs_filename)
        log.info(f"Using pre-computed Jacobians from {jacs_filename}.")
        log.info(
            f"Loaded {sum(j is not None for j in jacobians)}/{len(jacobians)} saved Jacobians."
        )
    except:
        logger.info("No pre-computed Jacobians found, generating from scratch...")
        jacobians = [None for _ in range(len(loader))]

    for i, batch in enumerate(tqdm(loader)):
        batch.split = split
        batch.to(torch.device(cfg.accelerator))
        batch_original = batch.clone().detach()
        num_nodes = batch_original.num_nodes - int(cfg.dataset.virtual_node)
        num_out_node = clamp_nodes(round(num_nodes * cfg.longrange.out_node_prob))
        num_in_node = round(
            num_nodes * cfg.longrange.in_node_prob
        )  # we don't clamp num_in_nodes — TODO remove num_in_nodes as a sampling param entirely, it is biased and provides no speedup for <1.

        x = batch_original.x
        device = x.device
        torch.manual_seed(cfg.longrange.sampling_seed)
        in_dim_mask = torch.randperm(x.shape[1])[:num_in_channel]
        out_dim_mask = torch.randperm(output_dim)[:num_out_channel]
        total_nodes = x.shape[0] - int(
            cfg.dataset.virtual_node
        )  # we don't care about the virtual node and don't want to sample it
        in_node_mask = torch.randperm(total_nodes)[:num_in_node]
        out_node_mask = torch.randperm(total_nodes)[:num_out_node]

        ## GET JACOBIAN ##

        if jacobians[i] is not None:
            # Load pre-saved Jacobian
            summed_abs_jacobian = jacobians[i]

            # Can remove this once the existing saved Jacobians are updated
            if summed_abs_jacobian.shape != (num_out_node, num_in_node):
                if summed_abs_jacobian.shape == (total_nodes, total_nodes):
                    logger.warning("Saved Jacobian is in full form rather than sampled form, which is wasteful of hard drive storage.")
                    # If the Jacobian is square already, slice it to get the correct shape and re-save
                    summed_abs_jacobian = summed_abs_jacobian[
                        out_node_mask[:, None], in_node_mask[None, :]
                    ]
                else:
                    raise ValueError(
                        f"Loaded Jacobian shape {summed_abs_jacobian.shape} does not match expected shape {(num_out_node, num_in_node)}."
                    )

        else:
            # Compute Jacobian from scratch according to specified method
            differentiable_model = (
                HeadlessGraphLevelModelWrapper(model) if use_prepooling_layer else model
            )  # remove final pooling layer of model if using node-level Jacobian for graph-level tasks

            diff_model = DiffModel(
                model=differentiable_model,
                data_batch=batch_original,
                output_dim=output_dim,
                lrgb_mode=True,
            )
            diff_model.gpu_tracker.start()

            ## COMPUTE/SAMPLE JACOBIAN ##
            summed_abs_jacobian = diff_model.sample_jacobian(
                x=x,
                in_node_mask=in_node_mask.to(device),
                in_dim_mask=in_dim_mask.to(device),
                out_node_mask=out_node_mask.to(device),
                out_dim_mask=out_dim_mask.to(device),
            ).cpu()  # [out_nodes, out_dim, in_node, in_dim]

            diff_model.gpu_tracker.end(i, logpath=gpu_stats_filename)
            jacobians[i] = summed_abs_jacobian
            del diff_model
            torch.cuda.empty_cache()

            # Save raw (abs, summed, sampled) Jacobians
            if i % 10 == 0:
                torch.save(jacobians, jacs_filename)
                log.info(f"Saved {i} Jacobians to {jacs_filename}")

        # Create a full Jacobian of zeros with flattened shape:
        jacobian = torch.zeros(total_nodes, total_nodes)
        # Insert the sampled values into the full Jacobian
        jacobian[
            out_node_mask[:, None],
            in_node_mask[None, :],
        ] = summed_abs_jacobian # -> [N x N]

        ## COMPUTE RANGE MEASURES ##
        for norm in [True, False]:
            for dist in ["res", "spd"]:
                rho = compute_range(
                    data=batch_original,
                    range_mat=jacobian,
                    norm=norm,
                    dist=dist,
                    out_dim=output_dim,
                    virtual_node=cfg.dataset.virtual_node,
                    **dict(
                        num_out_nodes=num_out_node,
                        num_in_nodes=num_in_node,
                        num_in_channels=num_in_channel,
                        num_out_channels=num_out_channel,
                    ),
                )
                range_stats[f"range_{dist}{'_norm' if norm else ''}"] += float(rho)
                range_vars[  # rho^2 for variance
                    f"range_{dist}{'_norm' if norm else ''}"
                ] += (float(rho) ** 2)

    ## AGGREGATE AND SAVE STATS ##

    # Get means of range measures
    for key in range_stats.keys():
        range_stats[key] /= len(loader)
    # Get variances of range measures
    for key in range_vars.keys():
        mean = range_stats[key]
        mean_sq = range_vars[key] / len(loader)
        range_vars[key] = mean_sq - mean**2

    # Save raw (abs, summed) Jacobians
    torch.save(jacobians, jacs_filename)
    log.info(f"Saved Jacobians to {jacs_filename}")

    # Save range stats
    range_df = pd.DataFrame({"range": range_stats, "var": range_vars})
    range_df.to_csv(ranges_filename)
    log.info(f"Saved range stats to {ranges_filename}")

    # No need to proceed with GraphGym run, we only want range stats
    sys.exit(0)