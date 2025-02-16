import torch
from torch_geometric.data import Data
from torch_geometric.datasets.graph_generator import ERGraph
from torch_geometric.utils import remove_self_loops
from loguru import logger
from tqdm import tqdm
import os


class ErdosDataset(object):
    """Generates Erdos-Renyi graphs with a given probability of edge creation"""

    def __init__(
        self,
        seed: int,
        size: int = 100,
        prob: float = 0.3,
        num_graphs: int = 500,
        transform=None,
        pre_transform=None,
    ):
        super().__init__()
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)

        self.pre_transform = pre_transform
        self.transform = transform

        self.size = size
        self.prob = prob
        self.num_graphs = num_graphs

        self._data = self.create_data()

    def get(self, i) -> Data:
        return self._data[i]

    def __getitem__(self, i) -> Data:
        """Redirects subscript calls (e.g. self[0]) to the `get` method"""
        return self.get(i)

    def create_data(self) -> Data:
        data_pre = []
        data_ls = []
        try:
            data_pre = torch.load(
                f"datasets/erdos/erdos_dataset_{self.size}_{self.prob}_{self.num_graphs}.pt"
            )
            logger.info("Loading precomputed Erdos dataset...")
            for i in tqdm(range(self.num_graphs)):
                data = data_pre[i]
                if self.transform is not None:
                    data = self.transform(data)
                data_ls += [data]
        except FileNotFoundError:
            logger.info("Generating Erdos dataset...")
            generator = ERGraph(self.size, self.prob)
            for _ in tqdm(range(self.num_graphs)):
                data = generator()
                data.edge_index = remove_self_loops(data.edge_index)[0]
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                    data_pre += [data]
                if self.transform is not None:
                    data = self.transform(data)
                data_ls += [data]
            os.makedirs("datasets/erdos", exist_ok=True)
            torch.save(
                data_pre,
                f"datasets/erdos/erdos_dataset_{self.size}_{self.prob}_{self.num_graphs}.pt",
            )
        return data_ls


if __name__ == "__main__":
    data = ErdosDataset(seed=0)
