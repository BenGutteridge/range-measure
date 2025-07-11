from argparse import Namespace
import torch
import sys
import tqdm
from typing import Tuple, Any, Callable, Optional
from torch_geometric.loader import DataLoader
import torch_geometric as pyg
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from loguru import logger
import wandb
from helpers.classes import GNNArgs, ActivationType
from helpers.metrics import LossesAndMetrics
from helpers.utils import set_seed
from models.MPNN import MPNN
from helpers.dataset_classes.dataset import DatasetBySplit
from longrange.task_preprocessing import (
    syn_task_preprocessing,
    syn_topology_preprocessing,
    add_distance_attrs,
)
from longrange.range import compute_range, DiffModel
import os
from os.path import join
import pandas as pd
import json


class Experiment(object):
    def __init__(
        self,
        args: Namespace,
    ):
        super().__init__()
        for arg in vars(args):
            value_arg = getattr(args, arg)
            logger.info(f"{arg}: {value_arg}")
            self.__setattr__(arg, value_arg)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_range = args.sample_range = any(
            [
                self.in_channel_sample_prob != 1.0,
                self.in_node_sample_prob != 1.0,
                self.out_channel_sample_prob != 1.0,
                self.out_node_sample_prob != 1.0,
            ]
        ) # if all sampling params are 1.0 (the default) then we don't sample
        set_seed(seed=self.seed)
        self.timestamp = time.strftime("%Y-%m-%d_%H%M")
        if self.wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                entity="Long-Range",
                project=self.wandb_project,
                name=f"{vars(args)['model_type']}.{vars(args)['num_layers']}",
                # track hyperparameters and run metadata
                config=vars(args),
            )
        # Define a directory to save models
        self.model_path = Path(self.model_path) if self.model_path is not None else None
        self.model_dir = Path("model_weights") / self.timestamp
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Parameters
        self.metric_type = self.dataset.get_metric_type()
        self.decimal = self.dataset.num_after_decimal()
        self.task_loss = self.metric_type.get_task_loss()
        self.graph_level = not self.dataset.is_node_based()

        # Synthetic LR task
        self.pre_transform = self.generate_synth_topology()
        
        if self.dataset.is_lr_synthetic():
            self.transform = self.generate_synth_task()
            self.dir_name = (
                "LR-SYNTH"
                f"_dist={self.distance_fn}"
                f"_int={self.interaction_fn}"
            ) + (f"_track_range" if self.track_range else "")
        else:
            self.transform = None
            self.dir_name = None

        if self.dataset.is_heterophilic() or self.track_range:

            def add_res_spd_dists(data):
                data = pyg.transforms.ToUndirected()(data)
                data = add_distance_attrs(data, "resistance")
                data = add_distance_attrs(data, "spd")
                return data

            self.pre_transform = add_res_spd_dists
            self.dir_name = "res_spd"

        # Asserts
        self.dataset.asserts(args)

    def run(self) -> Tuple[Tensor, Tensor]:
        dataset = self.dataset.load(
            seed=self.seed,
            pos_enc=self.pos_enc,
            pre_transform=self.pre_transform,
            transform=self.transform,
            dir_name=self.dir_name,
            subset=self.subset,
            num_graphs=self.num_graphs,
            virtual_node=self.virtual_node,
        )

        if self.metric_type.is_multilabel():
            dataset.data.y = dataset.data.y.to(dtype=torch.float)

        folds = self.dataset.get_folds(fold=self.fold)
        self.dataset_name = str(self.dataset)

        out_dim = self.metric_type.get_out_dim(dataset=dataset)
        gin_mlp_func = self.dataset.gin_mlp_func()  # only used if model_type=GIN
        act_type: ActivationType = self.dataset.activation_type()

        gnn_args = GNNArgs(
            model_type=self.model_type,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            layer_norm=self.layer_norm,
            skip=self.skip,
            batch_norm=self.batch_norm,
            dropout=self.dropout,
            act_type=act_type,
            metric_type=self.metric_type,
            in_dim=dataset[0].x.shape[1],
            out_dim=out_dim,
            gin_mlp_func=gin_mlp_func,
            dec_num_layers=self.dec_num_layers,
            pos_enc=self.pos_enc,
            dataset_encoders=self.dataset.get_dataset_encoders(),
            virtual_node=self.virtual_node,
        )

        metrics_list = []
        loss_histories = {"train": [], "val": [], "test": []}

        for num_fold in folds:
            set_seed(seed=self.seed)
            dataset_by_split = self.dataset.select_fold_and_split(
                num_fold=num_fold, dataset=dataset
            )

            if self.evaluate_only:
                if self.model_path is None:
                    # Construct the default model path if not provided
                    current_model_path = self.model_dir / f"fold_{num_fold}.pth"
                else:
                    current_model_path = self.model_path / f"fold_{num_fold}.pth"

                if not current_model_path.exists():
                    raise FileNotFoundError(
                        f"Model file not found at {current_model_path}"
                    )

                # Initialize and load the model
                model = MPNN(gnn_args=gnn_args, pool=self.pool).to(device=self.device)
                self.load_model(model=model, fold=num_fold, path=current_model_path)

                # Perform evaluation without training
                best_losses_n_metrics, fold_loss_history = self.evaluate_fold(
                    dataset_by_split=dataset_by_split,
                    model=model,
                    num_fold=num_fold,
                )
            else:
                # Perform training and evaluation
                best_losses_n_metrics, fold_loss_history = self.single_fold(
                    dataset_by_split=dataset_by_split,
                    gnn_args=gnn_args,
                    num_fold=num_fold,
                )

            # Print final
            print_str = f"Fold {num_fold}/{len(folds)}"
            for name in best_losses_n_metrics._fields:
                print_str += f",{name}={round(getattr(best_losses_n_metrics, name), self.decimal)}"
            logger.info(print_str)

            metrics_list.append(best_losses_n_metrics.get_fold_metrics())

            # Append loss histories
            for key in loss_histories:
                loss_histories[key].append(fold_loss_history[key])

        metrics_matrix = torch.stack(metrics_list, dim=0)  # (F, 3)
        metrics_mean = torch.mean(metrics_matrix, dim=0).tolist()  # (3,)

        # Logging
        logger.info(
            f"\nFinal train={round(metrics_mean[0], self.decimal)},"
            f"\nval={round(metrics_mean[1], self.decimal)},"
            f"\ntest={round(metrics_mean[2], self.decimal)}"
        )

        if len(folds) > 1:
            metrics_std = torch.std(metrics_matrix, dim=0).tolist()  # (3,)
            logger.info(
                f"\nFinal train={round(metrics_mean[0], self.decimal)}+-{round(metrics_std[0], self.decimal)},"
                f"\nval={round(metrics_mean[1], self.decimal)}+-{round(metrics_std[1], self.decimal)},"
                f"\ntest={round(metrics_mean[2], self.decimal)}+-{round(metrics_std[2], self.decimal)}"
            )

            run_scores_path = join(
                "range_results",
                str(self.dataset).replace("Dataset.", ""),
                str(self.model_type).replace("ModelType.", "") + ".json",
            )
            # Convert gnn_args and metrics_mean and _spd to a dict
            run_dict = {k: str(v) for k, v in gnn_args._asdict().items()} | {
                "train": metrics_mean[0],
                "val": metrics_mean[1],
                "test": metrics_mean[2],
                "train_std": metrics_std[0],
                "val_std": metrics_std[1],
                "test_std": metrics_std[2],
            }
            with open(run_scores_path, "a") as jsonfile:
                json.dump(run_dict, jsonfile, indent=2)
            logger.info(f"Saved run scores to {run_scores_path}")
            loss_histories_path = run_scores_path.replace(
                ".json", "_loss_histories.json"
            )
            with open(loss_histories_path, "a") as jsonfile:
                json.dump(loss_histories, jsonfile, indent=2)
            logger.info(f"Saved run scores to {loss_histories_path}")

        # Plot loss curves if training was performed
        # if not self.evaluate_only:
        #     self.plot_loss_curves(loss_histories)

        return metrics_mean

    def plot_loss_curves(
        self,
        loss_histories: dict,
        plot_dir: Path = Path("./plots"),
    ):
        """
        Plots the training, validation, and test loss curves for each fold.

        Args:
            loss_histories (dict): A dictionary containing lists of loss values for
                                   'train', 'val', and 'test' keys. Each key should
                                   map to a list of lists, where each inner list
                                   represents the loss values for a specific fold
                                   across epochs.

        Returns:
            None
        """
        plt.figure(figsize=(12, 8))
        epochs = range(self.max_epochs)

        for fold_idx, (train_loss, val_loss, test_loss) in enumerate(
            zip(loss_histories["train"], loss_histories["val"], loss_histories["test"])
        ):
            plt.plot(
                epochs,
                train_loss,
                label=f"Fold {fold_idx + 1} Train Loss",
                linestyle="--",
            )
            plt.plot(
                epochs, val_loss, label=f"Fold {fold_idx + 1} Val Loss", linestyle=":"
            )
            plt.plot(
                epochs, test_loss, label=f"Fold {fold_idx + 1} Test Loss", linestyle="-"
            )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curves Across Folds")
        plt.legend()
        plt.grid(True)
        filepath = plot_dir / f"{self.timestamp}_loss_curves.png"
        plt.savefig(filepath)

    def single_fold(
        self,
        dataset_by_split: DatasetBySplit,
        gnn_args: GNNArgs,
        num_fold: int,
    ) -> Tuple[LossesAndMetrics, dict]:
        model = MPNN(gnn_args=gnn_args, pool=self.pool).to(device=self.device)

        optimizer = self.dataset.optimizer(
            model=model, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = self.dataset.scheduler(
            optimizer=optimizer,
            step_size=self.step_size,
            gamma=self.gamma,
            num_warmup_epochs=self.num_warmup_epochs,
            max_epochs=self.max_epochs,
        )

        with tqdm.tqdm(total=self.max_epochs, file=sys.stdout) as pbar:
            logger.info(f"\nTraining on fold {num_fold}...")
            best_losses_n_metrics, loss_history = self.train_and_test(
                dataset_by_split=dataset_by_split,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                pbar=pbar,
                num_fold=num_fold,
            )

        return best_losses_n_metrics, loss_history

    def train_and_test(
        self,
        dataset_by_split: DatasetBySplit,
        model: MPNN,
        optimizer,
        scheduler: torch.optim.lr_scheduler,
        pbar: tqdm.tqdm,
        num_fold: int,
    ) -> Tuple[LossesAndMetrics, dict]:
        train_loader = DataLoader(
            dataset_by_split.train, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            dataset_by_split.val, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            dataset_by_split.test, batch_size=self.batch_size, shuffle=True
        )

        # separate loader of batch_size 1 for computing the jacobian and hessian to avoid memory blowup
        train_loader_range = DataLoader(
            dataset_by_split.train, batch_size=1, shuffle=True
        )
        val_loader_range = DataLoader(dataset_by_split.val, batch_size=1, shuffle=True)
        test_loader_range = DataLoader(
            dataset_by_split.test, batch_size=1, shuffle=True
        )

        best_losses_n_metrics = self.metric_type.get_worst_losses_n_metrics()
        loss_history = {
            "train": [],
            "val": [],
            "test": [],
            "test_range": [],
            "val_range": [],
            "train_range": [],
            "test_range_norm": [],
            "val_range_norm": [],
            "train_range_norm": [],
            "test_range_spd": [],
            "val_range_spd": [],
            "train_range_spd": [],
            "test_range_spd_norm": [],
            "val_range_spd_norm": [],
            "train_range_spd_norm": [],
        }

        # Task jacobian:
        if self.track_range and self.dataset.is_lr_synthetic():
            (
                train_task_range_res,
                train_task_range_spd,
                train_task_range_res_norm,
                train_task_range_spd_norm,
            ) = self.compute_task_range(loader=train_loader_range)
            (
                val_task_range_res,
                val_task_range_spd,
                val_task_range_res_norm,
                val_task_range_spd_norm,
            ) = self.compute_task_range(loader=val_loader_range)
            (
                test_task_range_res,
                test_task_range_spd,
                test_task_range_res_norm,
                test_task_range_spd_norm,
            ) = self.compute_task_range(loader=test_loader_range)
            print("VAL task range: ", val_task_range_res, val_task_range_res_norm)
            if self.wandb:
                wandb.log(
                    {
                        "Train/TaskRangeRes": train_task_range_res,
                        "Val/TaskRangeRes": val_task_range_res,
                        "Test/TaskRangeRes": test_task_range_res,
                        "Train/TaskRangeResNorm": train_task_range_res_norm,
                        "Val/TaskRangeResNorm": val_task_range_res_norm,
                        "Test/TaskRangeResNorm": test_task_range_res_norm,
                        "Train/TaskRangeSPD": train_task_range_spd,
                        "Val/TaskRangeSPD": val_task_range_spd,
                        "Test/TaskRangeSPD": test_task_range_spd,
                        "Train/TaskRangeSPDNorm": train_task_range_spd_norm,
                        "Val/TaskRangeSPDNorm": val_task_range_spd_norm,
                        "Test/TaskRangeSPDNorm": test_task_range_spd_norm,
                    }
                )
        for epoch in range(self.max_epochs):
            # Training step
            self.train(train_loader=train_loader, model=model, optimizer=optimizer)

            # Compute losses and metrics
            train_loss, train_metric = self.test(
                loader=train_loader, model=model, split_mask_name="train_mask"
            )
            if self.dataset.is_expressivity():
                val_loss, val_metric = train_loss, train_metric
                test_loss, test_metric = train_loss, train_metric
            else:
                val_loss, val_metric = self.test(
                    loader=val_loader, model=model, split_mask_name="val_mask"
                )
                test_loss, test_metric = self.test(
                    loader=test_loader, model=model, split_mask_name="test_mask"
                )

            if self.track_range and (
                epoch % self.track_epoch_every == 0
                or epoch == self.max_epochs - 1
                or epoch in self.track_epochs
            ):
                splits = self.range_splits
                ranges = {split: {} for split in splits}

                for split in splits:
                    if split == "train":
                        loader = train_loader_range
                    elif split == "val":
                        loader = val_loader_range
                    else:
                        loader = test_loader_range

                    # Compute range stats
                    ranges[split]["res"], ranges[split]["spd"], ranges[split][
                        "res_norm"
                    ], ranges[split]["spd_norm"] = self.compute_range_stats(
                        loader=loader,
                        model=model,
                        split_mask_name=f"{split}_mask",
                        epoch=epoch,
                        num_fold=num_fold,
                    )

                logger.info(f"Epoch {epoch}: Range stats: {ranges}")

            # Append to loss history
            loss_history["train"].append(train_loss)
            loss_history["val"].append(val_loss)
            loss_history["test"].append(test_loss)
            if self.track_range and epoch % self.track_epoch_every == 0:
                for split in splits:
                    range_res = ranges[split]["res"]
                    range_spd = ranges[split]["spd"]
                    range_res_norm = ranges[split]["res_norm"]
                    range_spd_norm = ranges[split]["spd_norm"]

                    # Append to loss history
                    loss_history[f"{split}_range"].append(range_res)
                    loss_history[f"{split}_range_spd"].append(range_spd)
                    loss_history[f"{split}_range_norm"].append(range_res_norm)
                    loss_history[f"{split}_range_spd_norm"].append(range_spd_norm)

            if self.wandb:
                wandb_log = {
                    "Train/Loss": train_loss,
                    "Val/Loss": val_loss,
                    "Test/Loss": test_loss,
                    f"Train/{self.metric_type}": train_metric,
                    f"Val/{self.metric_type}": val_metric,
                    f"Test/{self.metric_type}": test_metric,
                }
                if self.track_range and epoch % self.track_epoch_every == 0:
                    for split in splits:
                        wandb_log.update(
                            {
                                f"{split.capitalize()}/RangeRes": ranges[split]["res"],
                                f"{split.capitalize()}/RangeResNorm": ranges[split]["res_norm"],
                                f"{split.capitalize()}/RangeSPD": ranges[split]["spd"],
                                f"{split.capitalize()}/RangeSPDNorm": ranges[split]["spd_norm"],
                            }
                        )

                wandb.log(wandb_log)


            # Store current metrics
            losses_n_metrics = LossesAndMetrics(
                train_loss=train_loss,
                val_loss=val_loss,
                test_loss=test_loss,
                train_metric=train_metric,
                val_metric=val_metric,
                test_metric=test_metric,
            )

            # Step scheduler if applicable
            if scheduler is not None:
                scheduler.step(losses_n_metrics.val_metric)

            # Update best metrics if validation improves
            if self.metric_type.src_better_than_other(
                src=losses_n_metrics.val_metric, other=best_losses_n_metrics.val_metric
            ):
                best_losses_n_metrics = losses_n_metrics

                # Save the best model for the current fold
                model_path = self.model_dir / f"fold_{num_fold}.pth"
                torch.save(model.state_dict(), model_path)

            # Logging for progress
            log_str = f"Split: {num_fold}, Epoch: {epoch}"
            for name in losses_n_metrics._fields:
                log_str += (
                    f",{name}={round(getattr(losses_n_metrics, name), self.decimal)}"
                )
            log_str += f" (Best Test Metric: {round(best_losses_n_metrics.test_metric, self.decimal)})"
            pbar.set_description(log_str)
            pbar.update(1)

        return best_losses_n_metrics, loss_history

    def train(self, train_loader, model, optimizer):
        model.train()

        for data in train_loader:
            if self.batch_norm and (data.x.shape[0] == 1 or data.num_graphs == 1):
                continue
            optimizer.zero_grad()
            node_mask = self.dataset.get_split_mask(
                data=data, batch_size=data.num_graphs, split_mask_name="train_mask"
            ).to(self.device)
            edge_attr = data.edge_attr
            if data.edge_attr is not None:
                edge_attr = edge_attr.to(device=self.device)

            # forward
            scores = model(
                data.x.to(device=self.device),
                edge_index=data.edge_index.to(device=self.device),
                batch=data.batch.to(device=self.device),
                edge_attr=edge_attr,
                pestat=self.pos_enc.get_pe(data=data, device=self.device),
            )
            train_loss = self.task_loss(
                scores[node_mask], data.y.to(device=self.device)[node_mask]
            )

            # backward
            train_loss.backward(retain_graph=True)
            if self.dataset.clip_grad():
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

    def test(
        self,
        loader: DataLoader,
        model: MPNN,
        split_mask_name: str,
    ) -> Tuple[float, Any]:
        """
        Evaluate the model on the given data loader.

        Args:
            loader (DataLoader): The data loader containing the dataset to evaluate.
            model (MPNN): The model to be evaluated.
            split_mask_name (str): The name of the split mask to use for evaluation.

        Returns:
            Tuple[float, Any]: A tuple containing the average loss and the evaluation metric.
        """
        model.eval()

        total_loss = 0
        total_scores = np.empty(shape=(0, model.gnn_args.out_dim))
        total_y = None
        for data in loader:
            if self.batch_norm and (data.x.shape[0] == 1 or data.num_graphs == 1):
                continue
            node_mask = self.dataset.get_split_mask(
                data=data, batch_size=data.num_graphs, split_mask_name=split_mask_name
            ).to(device=self.device)
            edge_attr = data.edge_attr
            if data.edge_attr is not None:
                edge_attr = edge_attr.to(device=self.device)

            # forward
            scores = model(
                data.x.to(device=self.device),
                edge_index=data.edge_index.to(device=self.device),
                edge_attr=edge_attr,
                batch=data.batch.to(device=self.device),
                pestat=self.pos_enc.get_pe(data=data, device=self.device),
            )

            eval_loss = self.task_loss(scores, data.y.to(device=self.device))

            # analytics
            total_scores = np.concatenate(
                (total_scores, scores[node_mask].detach().cpu().numpy())
            )
            if total_y is None:
                total_y = (
                    data.y.to(device=self.device)[node_mask].detach().cpu().numpy()
                )
            else:
                total_y = np.concatenate(
                    (
                        total_y,
                        data.y.to(device=self.device)[node_mask].detach().cpu().numpy(),
                    )
                )

            total_loss += eval_loss.item() * data.num_graphs
        metric = self.metric_type.apply_metric(scores=total_scores, target=total_y)

        loss = total_loss / len(loader.dataset)
        return loss, metric

    def compute_range_stats(
        self,
        loader: DataLoader,
        model: MPNN,
        split_mask_name: str,
        **kwargs,
    ) -> Tuple[float, Any]:
        """
        Compute the Jacobian loss and the average range of the model on the given data loader.

        Args:
            loader (DataLoader): The data loader containing the dataset to evaluate,
            each data object contains a data.task_jacobian
            model (MPNN): The model to be evaluated.
            split_mask_name (str): The name of the split mask to use for evaluation.

        Returns:
            Tuple[float, Any]: A tuple containing the average jacobian metric, and the average range.
        """
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
        range_stats_path = None

        for data in loader:
            if self.batch_norm and (data.x.shape[0] == 1 or data.num_graphs == 1):
                continue # TODO: I'm not really sure what this is for, did we add it?
            data = data.to(device=self.device)

            if self.dataset.is_heterophilic() or len(data.y.shape) == 1:
                out_dim = len(torch.unique(data.y))
            else:
                out_dim = data.y.shape[1]

            # Compute Jacobian between model input and output
            diff_model = DiffModel(
                model=model,  # separate class to ensure safety wrt mutable vars
                data_batch=data,
                output_dim=out_dim,
                use_jacobian=self.use_jacobian,
            )

            # Next we compute the Jacobian of the model on the data, or the Hessian if graph_level
            sample_kwargs = {}
            if (
                self.graph_level and not self.use_jacobian
            ):  # if use_jacobian then we still compute jacobian even if graph-level
                hessian = diff_model.hessian(
                    data.x.to(device=self.device)
                ).detach()  # [num_graphs, out_dim, num_batch_nodes, in_dim, num_batch_nodes, in_dim] which squeezes(0) to [out_dim, num_batch_nodes, in_dim, num_batch_nodes, in_dim] when num_graphs=1
                range_mat = hessian.squeeze(
                    0
                )  # need batch_size==1  # if batch=1: [out_dim, num_batch_nodes, in_dim, num_batch_nodes, in_dim]]
                range_mat = range_mat.abs()
                range_mat = range_mat.sum(
                    0
                )  # [num_batch_nodes, in_dim, num_batch_nodes, in_dim] align the shape with that of node-level
                # Reduce Hessian along node feature channels
                # [N, out_dim, N, in_dim] -> [N, N]
                range_mat = torch.abs(range_mat)
                range_mat = range_mat.sum(dim=1).sum(dim=2)

            elif self.sample_range and "epoch" in kwargs:
                # Very large graphs necessitate a sampled Jacobian
                if len(loader)> 1:
                    raise NotImplementedError("Sampling only supported for single-graph datasets at present")
                epoch = kwargs["epoch"]
                num_fold = kwargs["num_fold"]
                split = split_mask_name.split("_")[0]
                dataset_str = str(self.dataset).replace("Dataset.", "")
                # Sampling stats
                in_channel_sample_prob = self.in_channel_sample_prob
                out_channel_sample_prob = self.out_channel_sample_prob
                in_node_sample_prob = self.in_node_sample_prob
                out_node_sample_prob = self.out_node_sample_prob
                sampling_seed = num_fold

                # Try to load saved Jacobian

                range_path = join(
                    "range_results",
                    dataset_str,
                    str(self.model_type).replace("ModelType.", ""),
                    split,
                    f"sampling-nodes-in{in_node_sample_prob:.2f}out{out_node_sample_prob:.2f}-channels-in{in_channel_sample_prob:.2f}out{out_channel_sample_prob:.2f}",
                    str(sampling_seed),
                )
                os.makedirs(join(range_path, str(sampling_seed)), exist_ok=True)
                range_stats_path = join(
                    range_path,
                    f"range_stats_fold{num_fold:02d}_e{epoch:03d}.csv",
                )
                if self.virtual_node:
                    range_stats_path = range_stats_path.replace(
                        "range_stats_", "range_stats_+vn_"
                    )
                jacobian_path = range_stats_path.replace("range_stats_", "").replace(
                    ".csv", ".pt"
                )

                total_nodes = data.x.shape[0] - int(self.virtual_node)
                num_in_channels = round(data.x.shape[1] * in_channel_sample_prob)
                num_out_channels = round(out_dim * out_channel_sample_prob)
                num_in_nodes = round(total_nodes * in_node_sample_prob)
                num_out_nodes = round(total_nodes * out_node_sample_prob)

                sample_kwargs = dict(
                    num_in_channels=num_in_channels,
                    num_out_channels=num_out_channels,
                    num_in_nodes=num_in_nodes,
                    num_out_nodes=num_out_nodes,
                )

                torch.manual_seed(sampling_seed)
                in_dim_mask = torch.randperm(data.x.shape[1])[:num_in_channels]
                out_dim_mask = torch.randperm(out_dim)[:num_out_channels]
                in_node_mask = torch.randperm(total_nodes)[:num_in_nodes]
                out_node_mask = torch.randperm(total_nodes)[:num_out_nodes]

                try:
                    jacobian = torch.load(
                        jacobian_path
                    )  # [num_out_nodes, num_in_nodes]
                    logger.info(f"Loaded Jacobian from {jacobian_path}")
                    assert (
                        jacobian.shape[0] == num_out_nodes
                    ), f"Shape mismatch, {jacobian.shape[0]=}, {num_out_nodes=}"
                except:

                    logger.info(
                        "Computing sampled Jacobian with the following parameters:"
                        f"num_in_channels: {num_in_channels}\n"
                        f"num_out_channels: {num_out_channels}\n"
                        f"num_in_nodes: {num_in_nodes}\n"
                        f"num_out_nodes: {num_out_nodes}\n"
                        f"sampling_seed: {sampling_seed}"
                    )
                    jacobian = diff_model.sample_jacobian(
                        data.x.to(device=self.device),
                        in_node_mask=in_node_mask,
                        out_node_mask=out_node_mask,
                        in_dim_mask=in_dim_mask,
                        out_dim_mask=out_dim_mask,
                    )  # -> abs summed matrix, num_out_nodes x num_in_nodes
                    torch.save(
                        jacobian, jacobian_path
                    )  # save this version to save space
                    logger.info(f"Saved Jacobian to {jacobian_path}")

                # Create a full (abs summed, so N x N) Jacobian of zeros
                range_mat = torch.zeros(total_nodes, total_nodes)
                # Insert the sampled values into the full Jacobian
                range_mat[
                    out_node_mask[:, None],
                    in_node_mask[None, :],
                ] = jacobian

            else:
                jacobian = diff_model.compute_jacobian(
                    data.x.to(device=self.device)
                )  # [num_batch_nodes, dim_out, num_batch_nodes, dim_in]
                range_mat = jacobian
                # Reduce Jacobian along node feature channels
                # [N, out_dim, N, in_dim] -> [N, N]
                range_mat = torch.abs(range_mat)
                range_mat = range_mat.sum(dim=1).sum(dim=2)

            ## COMPUTE RANGE MEASURES ##
            for norm in [True, False]:
                for dist in ["res", "spd"]:
                    if not hasattr(data, f"edge_attr_{dist}"):
                        logger.warning(
                            f"Data object does not have edge_attr_{dist}, skipping {dist} range measure"
                        )
                        continue
                    rho = compute_range(
                        data=data,
                        range_mat=range_mat,
                        norm=norm,
                        dist=dist,
                        out_dim=out_dim,
                        virtual_node=self.virtual_node,
                        **sample_kwargs,
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

        # Save range stats
        if range_stats_path is not None:
            range_df = pd.DataFrame({"range": range_stats, "var": range_vars})
            range_df.to_csv(range_stats_path)
            logger.info(f"Saved range stats to {range_stats_path}")

        return (
            range_stats["range_res"],
            range_stats["range_spd"],
            range_stats["range_res_norm"],
            range_stats["range_spd_norm"],
        )

    def compute_task_range(self, loader: DataLoader) -> Tuple[float, float]:
        """Compute ground-truth range measure for synthetic task using Jacobian and resistance/spd as range functions"""
        rho_res_total, rho_spd_total = 0, 0
        rho_res_norm_total, rho_spd_norm_total = 0, 0

        for data in loader:
            out_dim = data.y.shape[-1]
            in_dim = data.x.shape[-1]
            edge_index_dense = data.edge_index_dense.to(
                self.device
            )  # [2, batch_size*N**2]  of values ranging from 0 to batch_size*N

            if self.graph_level:
                task_hessian = torch.zeros(
                    [out_dim, data.num_nodes, in_dim, data.num_nodes, in_dim],
                    device=self.device,
                )
                edge_attr_hes = data.edge_attr_hes.to(
                    self.device
                )  # [N**2, dim_out, dimn_in, dim_in]
                task_hessian[:, edge_index_dense[0], :, edge_index_dense[1], :] = (
                    edge_attr_hes  # [dim_out, N, dim_in, N, dim_in]
                )
                task_hessian_abs = task_hessian.abs()
                range_mat = task_hessian_abs.sum(dim=0)  # [N, dim_in, N, dim_in]
            else:
                task_jacobian = torch.zeros(
                    [data.num_nodes, out_dim, data.num_nodes, in_dim],
                    device=self.device,
                )
                edge_attr_jac = data.edge_attr_jac.to(
                    self.device
                )  # [batch_size*N**2, out_dim, in_dim]
                task_jacobian[edge_index_dense[0], :, edge_index_dense[1], :] = (
                    edge_attr_jac
                )
                range_mat = task_jacobian  # [N, dim_in, N, dim_in]

            # Reduce Jacobian along node feature channels
            # [N, out_dim, N, in_dim] -> [N, N]
            range_mat_abs = torch.abs(range_mat)
            range_mat_abs = range_mat_abs.sum(dim=1).sum(dim=2)

            # Calculate range measures
            rho_res_total += compute_range(
                data.to(self.device),
                range_mat_abs,
                norm=False,
                dist="res",
                out_dim=out_dim,
            )

            rho_res_norm_total += compute_range(
                data.to(self.device),
                range_mat_abs,
                norm=True,
                dist="res",
                out_dim=out_dim,
            )

            rho_spd_total += compute_range(
                data.to(self.device),
                range_mat_abs,
                norm=False,
                dist="spd",
                out_dim=out_dim,
            )

            rho_spd_norm_total += compute_range(
                data.to(self.device),
                range_mat_abs,
                norm=True,
                dist="spd",
                out_dim=out_dim,
            )
        return (
            rho_res_total / len(loader.dataset),
            rho_spd_total / len(loader.dataset),
            rho_res_norm_total / len(loader.dataset),
            rho_spd_norm_total / len(loader.dataset),
        )

    def generate_synth_task(self) -> Optional[Callable]:
        """Returns a transform function that generates the LR synthetic task."""
        
        def transform(data):
            return syn_task_preprocessing(
                data=data,
                distance_fn=self.distance_fn,
                interaction_fn=self.interaction_fn,
                track_range=self.track_range,
                graph_level=self.graph_level,
            )

        return transform

    def generate_synth_topology(self) -> Optional[Callable]:
        """Returns a pre_transform function that generates the LR synthetic spd and resistance distances."""

        # Wrapper for syn_task_preprocessing with correct parameters
        def transform(data):
            return syn_topology_preprocessing(
                data=data,
            )

        return transform

    def load_model(self, model: MPNN, fold: int, path: Optional[Path] = None):
        """
        Loads the saved model weights for a specific fold.

        Args:
            model (MPNN): The model instance to load weights into.
            fold (int): The fold number whose model weights to load.
            path (Optional[Path]): Optional path to the model file. If not provided, uses the default path.

        Returns:
            MPNN: The model with loaded weights.
        """
        if path is None:
            path = self.model_dir / f"fold_{fold}_best_model.pth"
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        logger.info(f"Loaded model weights from {path}")
        return model

    def evaluate_fold(
        self,
        dataset_by_split: DatasetBySplit,
        model: MPNN,
        num_fold: int,
    ) -> Tuple[LossesAndMetrics, dict]:
        """
        Evaluates the loaded model on the specified fold without further training.

        Args:
            dataset_by_split (DatasetBySplit): The dataset splits for the current fold.
            model (MPNN): The pre-loaded model to evaluate.
            num_fold (int): The current fold number.

        Returns:
            Tuple[LossesAndMetrics, dict]: The best losses and metrics, and loss history.
        """
        # Initialize DataLoaders
        train_loader = DataLoader(
            dataset_by_split.train, batch_size=self.batch_size, shuffle=False
        )
        val_loader = DataLoader(
            dataset_by_split.val, batch_size=self.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            dataset_by_split.test, batch_size=self.batch_size, shuffle=False
        )

        loss_history = {"train": [], "val": [], "test": []}

        # Perform evaluation on all splits
        for split_name, loader in [
            ("train", train_loader),
            ("test", test_loader),
            ("val", val_loader),
        ]:
            loss, metric = self.test(
                loader=loader, model=model, split_mask_name=f"{split_name}_mask"
            )
            loss_history[split_name].append(loss)
            logger.info(
                f"Fold {num_fold} - {split_name.capitalize()} Loss: {round(loss, self.decimal)}, Metric: {round(metric, self.decimal)}"
            )

        # Aggregate metrics into LossesAndMetrics
        losses_n_metrics = LossesAndMetrics(
            train_loss=loss_history["train"][0],
            val_loss=loss_history["val"][0],
            test_loss=loss_history["test"][0],
            train_metric=metric if split_name == "train" else float("nan"),
            val_metric=metric if split_name == "val" else float("nan"),
            test_metric=metric if split_name == "test" else float("nan"),
        )

        return losses_n_metrics, loss_history
