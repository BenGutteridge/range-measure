from typing import Union, List
import torch
from torch_geometric.data import Data, Batch
from typing import Dict, Tuple, List
from torch import Tensor
from torch_geometric.datasets import StochasticBlockModelDataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import remove_self_loops


class SBMDataset(object):

    def __init__(
        self,
        seed: int,
        block_sizes: Union[List[int], Tensor] = [50, 50],
        edge_probs: Union[List[List[float]], Tensor] = [[0.75, 0.25], [0.25, 0.75]],
        num_graphs: int = 500,
        transform=None,
        pre_transform=None,
    ):
        super().__init__()
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)

        self.pre_transform = pre_transform
        self.transform = transform

        self.block_sizes = block_sizes
        self.edge_probs = edge_probs
        self.num_graphs = num_graphs

        self._data = self.create_data()

    def get(self, i) -> Data:
        return self._data[i]

    def __getitem__(self, i) -> Data:
        """Redirects subscript calls (e.g. self[0]) to the `get` method"""
        return self.get(i)

    def create_data(self) -> InMemoryDataset:
        # train, val, test
        dataset = StochasticBlockModelDataset(
            root="datasets",
            block_sizes=self.block_sizes,
            edge_probs=self.edge_probs,
            num_graphs=self.num_graphs,
            pre_transform=self.pre_transform,
            transform=self.transform,
            force_reload=True,
        )
        transf_dataset = [
            self.transform(dataset.get(i)) for i in range(self.num_graphs)
        ]  # the transform was not applied otherwise
        return transf_dataset


if __name__ == "__main__":
    data = SBMDataset(seed=0)
