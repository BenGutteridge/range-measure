import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
import pickle

from lrgb_exps.graphgps.loss.subtoken_prediction_loss import subtoken_cross_entropy
from lrgb_exps.graphgps.utils import (
    cfg_to_dict,
    flatten_dict,
    make_wandb_name,
    dirichlet_energy,
    mean_average_distance,
    mean_norm,
)
from longrange.range import compute_range, LRGBDiffModel, HeadlessGraphLevelModelWrapper
from loguru import logger as log
from tqdm import tqdm
import os
from os import path as osp
import sys
import pandas as pd


def train_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    for iter, batch in enumerate(loader):
        batch.split = "train"
        batch.to(torch.device(cfg.accelerator))
        pred, true = model(batch)
        if cfg.dataset.name == "ogbg-code2":
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to("cpu", non_blocking=True)
            _pred = pred_score.detach().to("cpu", non_blocking=True)

        loss.backward()
        # Parameters update after accumulating gradients for given num. batches.
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.optim.clip_grad_norm_value
                )
            optimizer.step()
            optimizer.zero_grad()

        logger.update_stats(
            true=_true,
            pred=_pred,
            loss=loss.detach().cpu().item(),
            lr=scheduler.get_last_lr()[0],
            time_used=time.time() - time_start,
            params=cfg.params,
            dataset_name=cfg.dataset.name,
        )
        time_start = time.time()


@torch.no_grad()
def eval_epoch(logger, loader, model, split="val"):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.accelerator))
        batch_original = batch.clone()
        if cfg.gnn.head == "inductive_edge":
            pred, true, extra_stats = model(batch)
        else:
            pred, true = model(batch)
            extra_stats = {}
        if cfg.dataset.name == "ogbg-code2":
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to("cpu", non_blocking=True)
            _pred = pred_score.detach().to("cpu", non_blocking=True)
        if cfg.train.eval_smoothing_metrics:
            extra_stats["dirichlet"] = dirichlet_energy(
                batch.x, batch.edge_index, batch.batch
            )
            extra_stats["mad"] = mean_average_distance(
                batch.x, batch.edge_index, batch.batch
            )
            extra_stats["emb_norm"] = mean_norm(batch.x)

        logger.update_stats(
            true=_true,
            pred=_pred,
            loss=loss.detach().cpu().item(),
            lr=0,
            time_used=time.time() - time_start,
            params=cfg.params,
            dataset_name=cfg.dataset.name,
            **extra_stats,
        )
        time_start = time.time()


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
    split = cfg.longrange.split
    subset = cfg.longrange.subset
    epoch = cfg.longrange.epoch
    load_ckpt(model, optimizer, scheduler, epoch)
    loader = {"train": loaders[0], "val": loaders[1], "test": loaders[2]}[split]

    file_id = f"{f'subset{subset}_' if subset > 0 else ''}e{epoch:03d}.pkl"
    ranges_filename = f"{cfg.run_dir}/{split}/range_stats_{file_id}"
    os.makedirs(f"{cfg.run_dir}/{split}/jacobians", exist_ok=True)
    jacs_filename = f"{cfg.run_dir}/{split}/jacobians/jacobians_{file_id}"

    if osp.exists(ranges_filename):
        log.info(f"Range stats already computed for split {split} and epoch {epoch}.")
        sys.exit(0)

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

    jacobians = [None for _ in range(len(loader))]
    if osp.exists(jacs_filename):
        log.info(f"Using pre-computed Jacobians from {jacs_filename}.")
        with open(jacs_filename, "rb") as f:
            jacobians = pickle.load(f)
            log.info(
                f"Loaded {sum(j is not None for j in jacobians)}/{len(jacobians)} saved Jacobians."
            )
    else:
        jacobians = [None for _ in range(len(loader))]

    for i, batch in enumerate(tqdm(loader)):

        batch.split = split
        batch.to(torch.device(cfg.accelerator))
        batch_original = batch.clone().detach()

        if jacobians[i] is not None:
            summed_abs_jacobian = jacobians[i]
        else:
            # Calculate Jacobian from scratch

            if (
                "peptides" in cfg.dataset.name and not use_hessian
            ):  # graph-level with Jacobian of node feats
                differentiable_model = HeadlessGraphLevelModelWrapper(model)
            else:
                differentiable_model = model

            diff_model = LRGBDiffModel(
                model=differentiable_model,
                data_batch=batch_original,
            )  # separate class to ensure safety wrt mutable vars

            if "peptides" in cfg.dataset.name:  # graph-level
                if use_hessian:
                    # Returns Hessian but we use Jacobian as catch-all for processing below
                    jacobian = (
                        diff_model.compute_hessian(batch_original.x).detach().cpu()
                    )
                    # Align the shape with that of node-level
                    jacobian = (
                        jacobian.abs().sum(1).squeeze(0)
                    )  # -> [num_graphs, num_batch_nodes, in_dim, num_batch_nodes, in_dim]
                else:  # use Jacobian on pre-pooling node features
                    # As first dims will be the same for input and output, stop jacobian function from treating it as a batch dim by adding a dummy one
                    x = batch_original.x.unsqueeze(0)

                    if diff_model.chunk_jacobian:
                        jacobian_abs_sum_chunks = []
                        for chunk_idx in range(diff_model.chunk_jacobian):
                            jacobian = (
                                diff_model.compute_jacobian(x, chunk_idx=chunk_idx)
                                .squeeze()
                                .detach()
                                .cpu()
                            )
                            jacobian_abs_sum_chunks.append(
                                jacobian.abs().sum(dim=1).sum(dim=2)
                            )
                            del jacobian
                            torch.cuda.empty_cache()
                        summed_abs_jacobian = torch.cat(jacobian_abs_sum_chunks, dim=0)
                    else:
                        jacobian = (
                            diff_model.compute_jacobian(x).squeeze().detach().cpu()
                        )
                        summed_abs_jacobian = jacobian.abs().sum(dim=1).sum(dim=2)
                        del jacobian
                        torch.cuda.empty_cache()

            elif cfg.dataset.format == "PyG-VOCSuperpixels":
                jacobian = (
                    diff_model.compute_jacobian(batch_original.x)
                    .squeeze()
                    .detach()
                    .cpu()
                )
                # Reduce Jacobian along node feature channels
                summed_abs_jacobian = (
                    torch.abs(jacobian).sum(dim=1).sum(dim=2)
                )  # [N, out_dim, N, in_dim] -> [N, N]
                del jacobian
            else:
                raise NotImplementedError(
                    f"Dataset {cfg.dataset.format} {cfg.dataset.name} not supported."
                )

            jacobians[i] = summed_abs_jacobian.cpu()
            del diff_model
            torch.cuda.empty_cache()

            # Save raw (abs, summed) Jacobians
            if i % 10 == 0:
                with open(jacs_filename, "wb") as f:
                    pickle.dump(jacobians, f)
                log.info(f"Saved {i} Jacobians to {jacs_filename}")

        for norm in [True, False]:
            for dist in ["res", "spd"]:
                rho = compute_range(
                    data=batch_original,
                    range_mat=summed_abs_jacobian,
                    norm=norm,
                    dist=dist,
                )
                range_stats[f"range_{dist}{'_norm' if norm else ''}"] += float(rho)
                range_vars[  # rho^2 for variance
                    f"range_{dist}{'_norm' if norm else ''}"
                ] += (float(rho) ** 2)

    # Get means of range measures
    for key in range_stats.keys():
        range_stats[key] /= len(loader)
    # Get variances of range measures
    for key in range_vars.keys():
        mean = range_stats[key]
        mean_sq = range_vars[key] / len(loader)
        range_vars[key] = mean_sq - mean**2

    # Save raw (abs, summed) Jacobians
    with open(jacs_filename, "wb") as f:
        pickle.dump(jacobians, f)
    log.info(f"Saved Jacobians to {jacs_filename}")

    # Save range stats
    range_df = pd.DataFrame({"range": range_stats, "var": range_vars})
    with open(ranges_filename, "wb") as f:
        pickle.dump(range_df, f)
    log.info(f"Saved range stats to {ranges_filename}")

    sys.exit(0)


@register_train("custom")
def custom_train(loggers, loaders, model, optimizer, scheduler):
    """
    Customized training pipeline.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler, cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info("Checkpoint found, Task already done")
    else:
        logging.info("Start from epoch %s", start_epoch)

    if cfg.wandb.use:
        try:
            import wandb
        except:
            raise ImportError("WandB is not installed.")
        if cfg.wandb.name == "":
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name
        run = wandb.init(
            entity=cfg.wandb.entity, project=cfg.wandb.project, name=wandb_name
        )
        run.config.update(cfg_to_dict(cfg))
        wandb.watch(model, log="all", log_freq=64)

    num_splits = len(loggers)
    split_names = ["val", "test"]
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        start_time = time.perf_counter()
        train_epoch(
            loggers[0],
            loaders[0],
            model,
            optimizer,
            scheduler,
            cfg.optim.batch_accumulation,
        )
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model, split=split_names[i - 1])
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == "reduce_on_plateau":
            scheduler.step(val_perf[-1]["loss"])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        # Checkpoint with regular frequency (if enabled).
        if (
            cfg.train.enable_ckpt
            and not cfg.train.ckpt_best
            and is_ckpt_epoch(cur_epoch)
        ):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch):
            best_epoch = np.array([vp["loss"] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""
            if cfg.metric_best != "auto":
                # Select again based on val perf of `cfg.metric_best`.
                m = cfg.metric_best
                best_epoch = getattr(
                    np.array([vp[m] for vp in val_perf]), cfg.metric_agg
                )()
                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    # Note: For some datasets it is too expensive to compute
                    # the main metric on the training set.
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                if cfg.wandb.use:
                    bstats = {"best/epoch": best_epoch}
                    for i, s in enumerate(["train", "val", "test"]):
                        bstats[f"best/{s}_loss"] = perf[i][best_epoch]["loss"]
                        if m in perf[i][best_epoch]:
                            bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                            run.summary[f"best_{s}_perf"] = perf[i][best_epoch][m]
                        for x in ["hits@1", "hits@3", "hits@10", "mrr"]:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                        for x in [
                            "hits@1_filt",
                            "hits@3_filt",
                            "hits@10_filt",
                            "mrr_filt",
                        ]:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                        for x in [
                            "hits@1_filt_self",
                            "hits@3_filt_self",
                            "hits@10_filt_self",
                            "mrr_filt_self",
                        ]:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                        for x in ["dirichlet", "mad", "emb_norm"]:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]

                    run.log(bstats, step=cur_epoch)
                    run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                    run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)
            # Checkpoint the best epoch params (if enabled).
            if (
                cfg.train.enable_ckpt
                and cfg.train.ckpt_best
                and best_epoch == cur_epoch
            ):
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                    clean_ckpt()
            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
            )
            if hasattr(model, "trf_layers"):
                # Log SAN's gamma parameter values if they are trainable.
                for li, gtl in enumerate(model.trf_layers):
                    if (
                        torch.is_tensor(gtl.attention.gamma)
                        and gtl.attention.gamma.requires_grad
                    ):
                        logging.info(
                            f"    {gtl.__class__.__name__} {li}: "
                            f"gamma={gtl.attention.gamma.item()}"
                        )
    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    # close wandb
    if cfg.wandb.use:
        run.finish()
        run = None

    logging.info("Task done, results saved in %s", cfg.run_dir)


@register_train("inference-only")
def inference_only(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference only.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    num_splits = len(loggers)
    split_names = ["train", "val", "test"]
    perf = [[] for _ in range(num_splits)]
    cur_epoch = 0
    start_time = time.perf_counter()

    for i in range(0, num_splits):
        eval_epoch(loggers[i], loaders[i], model, split=split_names[i])
        perf[i].append(loggers[i].write_epoch(cur_epoch))

    best_epoch = 0
    best_train = best_val = best_test = ""
    if cfg.metric_best != "auto":
        # Select again based on val perf of `cfg.metric_best`.
        m = cfg.metric_best
        if m in perf[0][best_epoch]:
            best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
        else:
            # Note: For some datasets it is too expensive to compute
            # the main metric on the training set.
            best_train = f"train_{m}: {0:.4f}"
        best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
        best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

    logging.info(
        f"> Inference | "
        f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
        f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
        f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
    )
    logging.info(f"Done! took: {time.perf_counter() - start_time:.2f}s")
    for logger in loggers:
        logger.close()


@register_train("PCQM4Mv2-inference")
def ogblsc_inference(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference on OGB-LSC PCQM4Mv2.

    Args:
        loggers: Unused, exists just for API compatibility
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    from ogb.lsc import PCQM4Mv2Evaluator

    evaluator = PCQM4Mv2Evaluator()

    num_splits = 3
    split_names = ["valid", "test-dev", "test-challenge"]
    assert len(loaders) == num_splits, "Expecting 3 particular splits."

    # Check PCQM4Mv2 prediction targets.
    logging.info(f"0 ({split_names[0]}): {len(loaders[0].dataset)}")
    assert all([not torch.isnan(d.y)[0] for d in loaders[0].dataset])
    logging.info(f"1 ({split_names[1]}): {len(loaders[1].dataset)}")
    assert all([torch.isnan(d.y)[0] for d in loaders[1].dataset])
    logging.info(f"2 ({split_names[2]}): {len(loaders[2].dataset)}")
    assert all([torch.isnan(d.y)[0] for d in loaders[2].dataset])

    model.eval()
    for i in range(num_splits):
        all_true = []
        all_pred = []
        for batch in loaders[i]:
            batch.to(torch.device(cfg.accelerator))
            pred, true = model(batch)
            all_true.append(true.detach().to("cpu", non_blocking=True))
            all_pred.append(pred.detach().to("cpu", non_blocking=True))
        all_true, all_pred = torch.cat(all_true), torch.cat(all_pred)

        if i == 0:
            input_dict = {"y_pred": all_pred.squeeze(), "y_true": all_true.squeeze()}
            result_dict = evaluator.eval(input_dict)
            logging.info(f"{split_names[i]}: MAE = {result_dict['mae']}")  # Get MAE.
        else:
            input_dict = {"y_pred": all_pred.squeeze()}
            evaluator.save_test_submission(
                input_dict=input_dict, dir_path=cfg.run_dir, mode=split_names[i]
            )


@register_train("log-attn-weights")
def log_attn_weights(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to inference on the test set and log the attention
    weights in Transformer modules.

    Args:
        loggers: Unused, exists just for API compatibility
        loaders: List of loaders
        model (torch.nn.Module): GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    import os.path as osp
    from torch_geometric.loader.dataloader import DataLoader
    from lrgb_exps.graphgps.utils import unbatch, unbatch_edge_index

    start_time = time.perf_counter()

    # The last loader is a test set.
    l = loaders[-1]
    # To get a random sample, create a new loader that shuffles the test set.
    loader = DataLoader(l.dataset, batch_size=l.batch_size, shuffle=True, num_workers=0)

    output = []
    # batch = next(iter(loader))  # Run one random batch.
    for b_index, batch in enumerate(loader):
        bsize = batch.batch.max().item() + 1  # Batch size.
        if len(output) >= 128:
            break
        print(f">> Batch {b_index}:")

        X_orig = unbatch(batch.x.cpu(), batch.batch.cpu())
        batch.to(torch.device(cfg.accelerator))
        model.eval()
        model(batch)

        # Unbatch to individual graphs.
        X = unbatch(batch.x.cpu(), batch.batch.cpu())
        edge_indices = unbatch_edge_index(batch.edge_index.cpu(), batch.batch.cpu())
        graphs = []
        for i in range(bsize):
            graphs.append(
                {
                    "num_nodes": len(X[i]),
                    "x_orig": X_orig[i],
                    "x_final": X[i],
                    "edge_index": edge_indices[i],
                    "attn_weights": [],  # List with attn weights in layers from 0 to L-1.
                }
            )

        # Iterate through GPS layers and pull out stored attn weights.
        for l_i, (name, module) in enumerate(model.model.layers.named_children()):
            if hasattr(module, "attn_weights"):
                print(l_i, name, module.attn_weights.shape)
                for g_i in range(bsize):
                    # Clip to the number of nodes in this graph.
                    # num_nodes = graphs[g_i]['num_nodes']
                    # aw = module.attn_weights[g_i, :num_nodes, :num_nodes]
                    aw = module.attn_weights[g_i]
                    graphs[g_i]["attn_weights"].append(aw.cpu())
        output += graphs

    logging.info(
        f"[*] Collected a total of {len(output)} graphs and their "
        f"attention weights for {len(output[0]['attn_weights'])} layers."
    )

    # Save the graphs and their attention stats.
    save_file = osp.join(cfg.run_dir, "graph_attn_stats.pt")
    logging.info(f"Saving to file: {save_file}")
    torch.save(output, save_file)

    logging.info(f"Done! took: {time.perf_counter() - start_time:.2f}s")
