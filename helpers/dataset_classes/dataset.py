import copy
import os.path as osp
from enum import Enum, auto
import torch
from torch import Tensor
from torch.utils.data import random_split
from typing import NamedTuple, Optional, List, Union, Callable
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.datasets import HeterophilousGraphDataset
import json
import numpy as np

from helpers.dataset_classes.root_neighbours_dataset import RootNeighboursDataset
from helpers.dataset_classes.cycles_dataset import CyclesDataset
from helpers.dataset_classes.lrgb import PeptidesFunctionalDataset, VOCSuperpixels
from helpers.dataset_classes.classic_datasets import Planetoid
from helpers.dataset_classes.grid_dataset import GridDataset
from helpers.dataset_classes.erdos import ErdosDataset
from helpers.dataset_classes.sbm import SBMDataset
from helpers.dataset_classes.barabasialbert import BADataset

from helpers.constants import ROOT_DIR
from helpers.metrics import MetricType
from helpers.classes import ActivationType, Pool, ModelType
from models.MPNN import MPNN
from helpers.encoders import DataSetEncoders, PosEncoder
from lrgb_synth.cosine_scheduler import cosine_with_warmup_scheduler
from lrgb_synth.transforms import apply_transform


class DatasetBySplit(NamedTuple):
    train: Union[Data, List[Data]]
    val: Union[Data, List[Data]]
    test: Union[Data, List[Data]]


class DataSetFamily(Enum):
    heterophilic = auto()
    synthetic = auto()
    social_networks = auto()
    proteins = auto()
    lrgb = auto()
    homophilic = auto()
    lr_synthetic = auto()
    lr_synthetic_graph = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return DataSetFamily[s]
        except KeyError:
            raise ValueError()


class Dataset(Enum):
    """
    an object for the different datasets
    """

    # heterophilic
    roman_empire = auto()
    amazon_ratings = auto()
    minesweeper = auto()
    tolokers = auto()
    questions = auto()

    # synthetic
    root_neighbours = auto()
    cycles = auto()

    gridgraphs = auto()
    linegraphs = auto()
    erdos = auto()
    erdossparse = auto()
    sbm = auto()
    sbmsparse = auto()
    ba = auto()

    gridgraphs_graph = auto()
    linegraphs_graph = auto()
    erdos_graph = auto()
    erdossparse_graph = auto()
    sbm_graph = auto()
    sbmsparse_graph = auto()
    ba_graph = auto()

    # social networks
    imdb_binary = auto()
    imdb_multi = auto()
    reddit_binary = auto()
    reddit_multi = auto()

    # proteins
    enzymes = auto()
    proteins = auto()
    nci1 = auto()

    # lrgb
    lrgb_func = auto()
    lrgb_voc = auto()

    # homophilic
    cora = auto()
    pubmed = auto()

    # LR synthetic tasks (2024-11)
    lrgb_func_synth = auto()
    lrgb_func_synth_graph = auto()
    lrgb_voc_synth = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return Dataset[s]
        except KeyError:
            raise ValueError()

    def get_family(self) -> DataSetFamily:
        if self in [
            Dataset.roman_empire,
            Dataset.amazon_ratings,
            Dataset.minesweeper,
            Dataset.tolokers,
            Dataset.questions,
        ]:
            return DataSetFamily.heterophilic
        elif self in [Dataset.root_neighbours, Dataset.cycles]:
            return DataSetFamily.synthetic
        elif self in [
            Dataset.imdb_binary,
            Dataset.imdb_multi,
            Dataset.reddit_binary,
            Dataset.reddit_multi,
        ]:
            return DataSetFamily.social_networks
        elif self in [Dataset.enzymes, Dataset.proteins, Dataset.nci1]:
            return DataSetFamily.proteins
        elif self in [Dataset.lrgb_func, Dataset.lrgb_voc]:
            return DataSetFamily.lrgb
        elif self in [Dataset.cora, Dataset.pubmed]:
            return DataSetFamily.homophilic
        elif self in [
            Dataset.lrgb_func_synth,
            Dataset.lrgb_voc_synth,
            Dataset.gridgraphs,
            Dataset.linegraphs,
            Dataset.erdos,
            Dataset.sbm,
            Dataset.sbmsparse,
            Dataset.erdossparse,
            Dataset.ba,
        ]:
            return DataSetFamily.lr_synthetic
        elif self in [
            Dataset.lrgb_func_synth_graph,
            Dataset.gridgraphs_graph,
            Dataset.linegraphs_graph,
            Dataset.erdos_graph,
            Dataset.sbm_graph,
            Dataset.sbmsparse_graph,
            Dataset.erdossparse_graph,
            Dataset.ba_graph,
        ]:
            return DataSetFamily.lr_synthetic_graph
        else:
            raise ValueError(f"Dataset {self.name} not supported in dataloader")

    def is_node_based(self) -> bool:
        return (
            self.get_family()
            in [
                DataSetFamily.heterophilic,
                DataSetFamily.homophilic,
                DataSetFamily.lr_synthetic,
            ]
            or self is Dataset.root_neighbours
            or self is Dataset.gridgraphs
            or self is Dataset.linegraphs
            or self is Dataset.erdos
            or self is Dataset.sbm
            or self is Dataset.sbmsparse
            or self is Dataset.erdossparse
            or self is Dataset.ba
            or self is Dataset.lrgb_voc
        )

    def not_synthetic(self) -> bool:
        return not (
            self.get_family()
            in [
                DataSetFamily.synthetic,
                DataSetFamily.lr_synthetic,
                DataSetFamily.lr_synthetic_graph,
            ]
        )

    def is_expressivity(self) -> bool:
        return self is Dataset.cycles

    def clip_grad(self) -> bool:
        return self.get_family() is DataSetFamily.lrgb

    def get_dataset_encoders(self):
        if self.get_family() in [
            DataSetFamily.heterophilic,
            DataSetFamily.synthetic,
            DataSetFamily.social_networks,
            DataSetFamily.proteins,
            DataSetFamily.homophilic,
            DataSetFamily.lr_synthetic,
            DataSetFamily.lr_synthetic_graph,
        ]:
            return DataSetEncoders.NONE
        elif self is Dataset.lrgb_func:
            return DataSetEncoders.MOL
        elif self is Dataset.lrgb_voc:
            return DataSetEncoders.VOC
        else:
            raise ValueError(
                f"Dataset {self.name} not supported in get_dataset_encoders"
            )

    def get_folds(self, fold: Optional[int]) -> List[int]:
        """
        Returns the list of fold indices based on the dataset family.

        Parameters:
        fold (int): The fold index to be used for certain dataset families.

        Returns:
        List[int]: A list of fold indices.

        Notes:
        - For heterophilic and homophilic dataset families, there are 10 folds for cross-validation
        - For all other datasets either a fold is provided or a single fold is used
        """
        if self.get_family() in [
            DataSetFamily.synthetic,
            DataSetFamily.lrgb,
            DataSetFamily.lr_synthetic,
            DataSetFamily.lr_synthetic_graph,
        ]:
            return list(range(1))
        elif self.get_family() in [
            DataSetFamily.heterophilic,
            DataSetFamily.homophilic,
        ]:
            return list(range(10))
        elif self.get_family() in [
            DataSetFamily.social_networks,
            DataSetFamily.proteins,
        ]:
            return [fold]
        else:
            raise ValueError(f"Dataset {self.name} not supported in dataloader")

    def load(
        self,
        seed: int,
        pos_enc: PosEncoder,
        transform: Optional[Callable],
        pre_transform: Optional[Callable],
        dir_name: Optional[str] = None,
        subset: Optional[int] = None,
        num_graphs: Optional[int] = None,
    ) -> List[Data]:
        root = osp.join(ROOT_DIR, "datasets")
        if self.get_family() is DataSetFamily.heterophilic:
            name = self.name.replace("_", "-").capitalize()
            dataset = [
                HeterophilousGraphDataset(
                    root=root, name=name, transform=T.ToUndirected()
                )[0]
            ]
        elif self.get_family() in [
            DataSetFamily.social_networks,
            DataSetFamily.proteins,
        ]:
            tu_dataset_name = self.name.upper().replace("_", "-")
            root = osp.join(ROOT_DIR, "datasets", tu_dataset_name)
            dataset = torch.load(root + ".pt")
        elif self is Dataset.root_neighbours:
            dataset = [RootNeighboursDataset(seed=seed).get()]
        elif self is Dataset.cycles:
            dataset = CyclesDataset().data
        elif self is Dataset.lrgb_func:
            dataset = PeptidesFunctionalDataset(root=root, subset=subset)
            dataset = apply_transform(dataset=dataset, pos_encoder=pos_enc)
        elif self in [Dataset.lrgb_func_synth, Dataset.lrgb_func_synth_graph]:
            dataset = PeptidesFunctionalDataset(
                root=root,
                pre_transform=pre_transform,
                transform=transform,
                dir_name=dir_name,
                subset=subset,
            )
            dataset = apply_transform(dataset=dataset, pos_encoder=pos_enc)
        elif self in [Dataset.lrgb_voc, Dataset.lrgb_voc_synth]:
            dataset_dir = osp.join(root, "vocsuperpixels")
            if self.get_family() is DataSetFamily.lr_synthetic:
                dataset_dir = osp.join(dataset_dir, dir_name)
            else:
                pre_transform = None
            if subset is not None:
                dataset_dir = osp.join(dataset_dir, f"subset={subset}")
            dataset = [
                apply_transform(
                    VOCSuperpixels(
                        root=dataset_dir,
                        subset=subset,
                        split=split,
                        pre_transform=pre_transform,
                    ),
                    pos_encoder=pos_enc,
                )
                for split in ["train", "val", "test"]
            ]
        elif self in [
            Dataset.gridgraphs,
            Dataset.gridgraphs_graph,
            Dataset.linegraphs,
            Dataset.linegraphs_graph,
        ]:
            if self in [Dataset.linegraphs, Dataset.linegraphs_graph]:
                height, width = 16, 1
            else:
                height, width = 16, 16
            dataset = GridDataset(
                seed=seed,
                transform=transform,
                pre_transform=pre_transform,
                height=height,
                width=width,
                num_graphs=num_graphs,
            )
        elif self in [
            Dataset.erdos,
            Dataset.erdossparse,
            Dataset.erdos_graph,
            Dataset.erdossparse_graph,
        ]:
            prob = (
                0.1 if self in [Dataset.erdossparse, Dataset.erdossparse_graph] else 0.3
            )
            dataset = ErdosDataset(
                seed=seed,
                pre_transform=pre_transform,
                transform=transform,
                size=100,
                prob=prob,
                num_graphs=num_graphs,
            )
        elif self in [
            Dataset.sbm,
            Dataset.sbmsparse,
            Dataset.sbm_graph,
            Dataset.sbmsparse_graph,
        ]:
            edge_probs = (
                [[0.3, 0.1], [0.1, 0.3]]
                if self in [Dataset.sbmsparse, Dataset.sbmsparse_graph]
                else [[0.75, 0.25], [0.25, 0.75]]
            )
            dataset = SBMDataset(
                seed=seed,
                pre_transform=pre_transform,
                transform=transform,
                block_sizes=[50, 50],
                edge_probs=edge_probs,
                num_graphs=num_graphs,
            )
        elif self in [Dataset.ba, Dataset.ba_graph]:
            dataset = BADataset(
                seed=seed,
                pre_transform=pre_transform,
                transform=transform,
                size=100,
                num_edges=5,
                num_graphs=num_graphs,
            )
        elif self.get_family() is DataSetFamily.homophilic:
            dataset = [
                Planetoid(root=root, name=self.name, transform=T.NormalizeFeatures())[0]
            ]
        else:
            raise ValueError(f"Dataset {self.name} not supported in dataloader")
        return dataset

    def select_fold_and_split(
        self, dataset: List[Data], num_fold: int
    ) -> DatasetBySplit:
        if self.get_family() is DataSetFamily.heterophilic:
            dataset_copy = copy.deepcopy(dataset)
            dataset_copy[0].train_mask = dataset_copy[0].train_mask[:, num_fold]
            dataset_copy[0].val_mask = dataset_copy[0].val_mask[:, num_fold]
            dataset_copy[0].test_mask = dataset_copy[0].test_mask[:, num_fold]
            return DatasetBySplit(
                train=dataset_copy, val=dataset_copy, test=dataset_copy
            )
        elif self.get_family() is DataSetFamily.synthetic:
            raise NotImplementedError(
                "Code below flagged as possibly erroneous, do not use until further review."
            )
            return DatasetBySplit(train=dataset, val=dataset, test=dataset)
        elif self.get_family() in [
            DataSetFamily.social_networks,
            DataSetFamily.proteins,
        ]:
            tu_dataset_name = self.name.upper().replace("_", "-")
            original_fold_dict = json.load(
                open(f"folds/{tu_dataset_name}_splits.json", "r")
            )[num_fold]
            model_selection_dict = original_fold_dict["model_selection"][0]
            split_dict = {
                "train": model_selection_dict["train"],
                "val": model_selection_dict["validation"],
                "test": original_fold_dict["test"],
            }
            dataset_by_splits = [
                [dataset[idx] for idx in split_dict[split]]
                for split in DatasetBySplit._fields
            ]
            return DatasetBySplit(*dataset_by_splits)
        elif self in [Dataset.lrgb_func, Dataset.lrgb_func_synth]:
            split_idx = dataset.get_idx_split()
            dataset_by_splits = [
                [dataset[idx] for idx in split_idx[split]]
                for split in DatasetBySplit._fields
            ]
            return DatasetBySplit(*dataset_by_splits)
        elif self is Dataset.lrgb_voc:
            return DatasetBySplit(*dataset)
        elif self in [
            Dataset.gridgraphs,
            Dataset.linegraphs,
            Dataset.erdos,
            Dataset.erdossparse,
            Dataset.sbm,
            Dataset.sbmsparse,
            Dataset.ba,
            Dataset.gridgraphs_graph,
            Dataset.linegraphs_graph,
            Dataset.erdos_graph,
            Dataset.erdossparse_graph,
            Dataset.sbm_graph,
            Dataset.sbmsparse_graph,
            Dataset.ba_graph,
        ]:
            num_data_train = int(dataset.num_graphs * 0.5)
            num_data_val = int(dataset.num_graphs * 0.25)
            train_data = dataset._data[:num_data_train]
            val_data = dataset._data[num_data_train : num_data_train + num_data_val]
            test_data = dataset._data[num_data_train + num_data_val :]
            return DatasetBySplit(train=train_data, val=val_data, test=test_data)
        elif self.get_family() is DataSetFamily.homophilic:
            device = dataset[0].x.device
            with np.load(
                f"folds/{self.name}_split_0.6_0.2_{num_fold}.npz"
            ) as folds_file:
                train_mask = torch.tensor(
                    folds_file["train_mask"], dtype=torch.bool, device=device
                )
                val_mask = torch.tensor(
                    folds_file["val_mask"], dtype=torch.bool, device=device
                )
                test_mask = torch.tensor(
                    folds_file["test_mask"], dtype=torch.bool, device=device
                )

            setattr(dataset[0], "train_mask", train_mask)
            setattr(dataset[0], "val_mask", val_mask)
            setattr(dataset[0], "test_mask", test_mask)

            dataset[0].train_mask[dataset[0].non_valid_samples] = False
            dataset[0].test_mask[dataset[0].non_valid_samples] = False
            dataset[0].val_mask[dataset[0].non_valid_samples] = False
            raise NotImplementedError(
                "Code below flagged as possibly erroneous, do not use until further review."
            )
            return DatasetBySplit(train=dataset, val=dataset, test=dataset)
        else:
            raise ValueError(f"NotImplemented")

    def get_metric_type(self) -> MetricType:
        if self.get_family() in [
            DataSetFamily.social_networks,
            DataSetFamily.proteins,
            DataSetFamily.homophilic,
        ] or self in [
            Dataset.roman_empire,
            Dataset.amazon_ratings,
            Dataset.cycles,
        ]:
            return MetricType.ACCURACY
        elif self in [Dataset.minesweeper, Dataset.tolokers, Dataset.questions]:
            return MetricType.AUC_ROC
        elif self in [
            Dataset.root_neighbours,
            Dataset.lrgb_func_synth,
            Dataset.lrgb_voc_synth,
            Dataset.gridgraphs,
            Dataset.linegraphs,
            Dataset.erdos,
            Dataset.erdossparse,
            Dataset.sbm,
            Dataset.sbmsparse,
            Dataset.ba,
            Dataset.lrgb_func_synth_graph,
            Dataset.gridgraphs_graph,
            Dataset.linegraphs_graph,
            Dataset.erdos_graph,
            Dataset.erdossparse_graph,
            Dataset.sbm_graph,
            Dataset.sbmsparse_graph,
            Dataset.ba_graph,
        ]:
            return MetricType.MSE_MAE
        elif self is Dataset.lrgb_func:
            return MetricType.MULTI_LABEL_AP
        elif self is Dataset.lrgb_voc:
            return MetricType.F1_MULTICLASS
        else:
            raise ValueError(f"Dataset {self.name} not supported in dataloader")

    def num_after_decimal(self) -> int:
        return 4 if self.get_family() is DataSetFamily.lrgb else 2

    def activation_type(self) -> ActivationType:
        """Determines the activation type (GELU/RELU) based on the dataset family."""
        if self.get_family() in [DataSetFamily.heterophilic, DataSetFamily.lrgb]:
            return ActivationType.GELU
        elif self.get_family() is DataSetFamily.lr_synthetic:
            return ActivationType.ID
        elif self.get_family() is DataSetFamily.lr_synthetic_graph:
            return (
                ActivationType.GELU  # need non-linearity since these are non-linear task with non-zero Jacobian
            )  # The Hessian of linear models is always 0, so should need to change this for the non-linear graph-level tasks
        else:
            return ActivationType.RELU

    def gin_mlp_func(self) -> Callable:
        """
        Generates a multi-layer perceptron (MLP) function for the GIN MPNN based on dataset type.
        Returns:
            Callable: A function that creates an MLP with specified input and output channels and bias.
        """

        if self in [
            Dataset.lrgb_func,
            Dataset.lrgb_func_synth,
            Dataset.lrgb_voc,
            Dataset.lrgb_voc_synth,
        ]:
            # 2 MLP layers
            def mlp_func(in_channels: int, out_channels: int, bias: bool):
                return torch.nn.Sequential(
                    torch.nn.Linear(in_channels, out_channels, bias=bias),
                    torch.nn.ReLU(),
                    torch.nn.Linear(out_channels, out_channels, bias=bias),
                )

        elif self.get_family() in [
            DataSetFamily.social_networks,
            DataSetFamily.proteins,
        ]:  # 2 MLP layers with double-width output channels

            def mlp_func(in_channels: int, out_channels: int, bias: bool):
                return torch.nn.Sequential(
                    torch.nn.Linear(in_channels, 2 * in_channels, bias=bias),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * in_channels, out_channels, bias=bias),
                )

        else:
            # 2 MLP layers with BN
            def mlp_func(in_channels: int, out_channels: int, bias: bool):
                return torch.nn.Sequential(
                    torch.nn.Linear(in_channels, 2 * in_channels, bias=bias),
                    torch.nn.BatchNorm1d(2 * in_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * in_channels, out_channels, bias=bias),
                )

        return mlp_func

    def optimizer(
        self,
        model: MPNN,
        lr: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        if self.get_family() in [
            DataSetFamily.heterophilic,
            DataSetFamily.synthetic,
            DataSetFamily.social_networks,
            DataSetFamily.proteins,
            DataSetFamily.homophilic,
            DataSetFamily.lr_synthetic,
            DataSetFamily.lr_synthetic_graph,
        ]:
            return torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif self.get_family() is DataSetFamily.lrgb:
            return torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Dataset {self.name} not supported in dataloader")

    def scheduler(
        self,
        optimizer,
        step_size: Optional[int],
        gamma: Optional[float],
        num_warmup_epochs: Optional[int],
        max_epochs: int,
    ):
        if self.get_family() is DataSetFamily.lrgb:
            assert (
                num_warmup_epochs is not None
            ), "cosine_with_warmup_scheduler's num_warmup_epochs is None"
            assert (
                max_epochs is not None
            ), "cosine_with_warmup_scheduler's max_epochs is None"
            return cosine_with_warmup_scheduler(
                optimizer=optimizer,
                num_warmup_epochs=num_warmup_epochs,
                max_epoch=max_epochs,
            )
        elif self.get_family() in [
            DataSetFamily.social_networks,
            DataSetFamily.proteins,
        ]:
            assert step_size is not None, "StepLR's step_size is None"
            assert gamma is not None, "StepLR's gamma is None"
            return torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer, step_size=step_size, gamma=gamma
            )
        elif self.get_family() in [
            DataSetFamily.heterophilic,
            DataSetFamily.synthetic,
            DataSetFamily.homophilic,
            DataSetFamily.lr_synthetic,
            DataSetFamily.lr_synthetic_graph,
        ]:
            return None
        else:
            raise ValueError(f"Dataset {self.name} not supported in dataloader")

    def get_split_mask(
        self, data: Data, batch_size: int, split_mask_name: str
    ) -> Tensor:
        if hasattr(data, split_mask_name):
            return getattr(data, split_mask_name)
        elif self.is_node_based():
            return torch.ones(size=(data.x.shape[0],), dtype=torch.bool)
        else:
            return torch.ones(size=(batch_size,), dtype=torch.bool)

    def asserts(self, args):
        # model
        assert (
            not (self.is_node_based()) or args.pool is Pool.NONE
        ), "Node based datasets have no pooling"
        assert (
            not (self.is_node_based()) or args.batch_norm is False
        ), "Node based dataset cannot have batch norm"
        assert not (
            not (self.is_node_based()) and args.pool is Pool.NONE
        ), "Graph based datasets need pooling"
        assert (
            args.model_type is not ModelType.LIN
        ), "The GNN layer type can't be linear"

        # dataset dependant parameters
        assert (
            self.get_family() in [DataSetFamily.social_networks, DataSetFamily.proteins]
            or args.fold is None
        ), "social networks and protein datasets are the only ones to use fold"
        assert (
            self.get_family()
            not in [DataSetFamily.social_networks, DataSetFamily.proteins]
            or args.fold is not None
        ), "social networks and protein datasets must specify fold"
        assert (
            self.get_family() is DataSetFamily.proteins
            or self.get_family() is DataSetFamily.social_networks
            or (args.step_size is None and args.gamma is None)
        ), "proteins datasets are the only ones to use step_size and gamma"
        assert self.get_family() is DataSetFamily.lrgb or (
            args.num_warmup_epochs is None
        ), "lrgb datasets are the only ones to use num_warmup_epochs"
        # encoders
        assert self.get_family() is DataSetFamily.lrgb or (
            args.pos_enc is PosEncoder.NONE
        ), "lrgb datasets are the only ones to use pos_enc"
