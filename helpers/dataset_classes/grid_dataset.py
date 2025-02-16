import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, from_networkx
from tqdm import tqdm
from networkx.generators import grid_2d_graph
from loguru import logger
import os


class GridDataset(object):

    def __init__(
        self,
        seed: int,
        height: int = 16,
        width: int = 16,
        num_graphs: int = 500,
        transform=None,
        pre_transform=None,
    ):
        super().__init__()
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)

        self.pre_transform = pre_transform
        self.transform = transform

        self.height = height
        self.width = width
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
                f"datasets/grid/grid_dataset_{self.height}_{self.width}_{self.num_graphs}.pt"
            )
            logger.info("Loading precomputed grid dataset...")
            for i in tqdm(range(self.num_graphs)):
                data = data_pre[i]
                if self.transform is not None:
                    data = self.transform(data)
                data_ls += [data]
        except FileNotFoundError:
            logger.info("Generating grid dataset...")
            for _ in tqdm(range(self.num_graphs)):
                graph = grid_2d_graph(self.height, self.width)
                data = from_networkx(graph)
                data.edge_index = remove_self_loops(data.edge_index)[0]
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                    data_pre += [data]

                if self.transform is not None:
                    data = self.transform(data)
                data_ls += [data]
            os.makedirs("datasets/grid", exist_ok=True)
            torch.save(
                data_pre,
                f"datasets/grid/grid_dataset_{self.height}_{self.width}_{self.num_graphs}.pt",
            )
        return data_ls


if __name__ == "__main__":
    data = GridDataset(seed=0)
