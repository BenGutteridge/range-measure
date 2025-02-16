import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (
    register_node_encoder,
    register_edge_encoder,
)
import torch_geometric as pyg
from typing import Union

"""
=== Description of the VOCSuperpixels dataset === 
Each graph is a tuple (x, edge_attr, edge_index, y)
Shape of x : [num_nodes, 14]
Shape of edge_attr : [num_edges, 1] or [num_edges, 2]
Shape of edge_index : [2, num_edges]
Shape of y : [num_nodes]
"""

VOC_node_input_dim = 14
# VOC_edge_input_dim = 1 or 2; defined in class VOCEdgeEncoder


@register_node_encoder("VOCNode")
class VOCNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        # Normalization stats: not in original LRGB, added in 'Where Did the Gap Go' (Tonshoff, 2023)
        node_x_mean = torch.tensor(
            [
                4.5824501e-01,
                4.3857411e-01,
                4.0561178e-01,
                6.7938097e-02,
                6.5604292e-02,
                6.5742709e-02,
                6.5212941e-01,
                6.2894762e-01,
                6.0173863e-01,
                2.7769071e-01,
                2.6425251e-01,
                2.3729359e-01,
                1.9344997e02,
                2.3472206e02,
            ]
        )
        node_x_std = torch.tensor(
            [
                2.5952947e-01,
                2.5716761e-01,
                2.7130592e-01,
                5.4822665e-02,
                5.4429270e-02,
                5.4474957e-02,
                2.6238337e-01,
                2.6600540e-01,
                2.7750680e-01,
                2.5197381e-01,
                2.4986187e-01,
                2.6069802e-01,
                1.1768297e02,
                1.4007195e02,
            ]
        )
        self.register_buffer("node_x_mean", node_x_mean)
        self.register_buffer("node_x_std", node_x_std)
        self.encoder = torch.nn.Linear(VOC_node_input_dim, emb_dim)

    def forward(
        self,
        input: Union[pyg.data.Batch, torch.Tensor],
        use_normalization: bool = True,
    ) -> Union[pyg.data.Batch, torch.Tensor]:
        """
        Forward method to handle either a Batch object or a Tensor object.

        Args:
            input (Union[torch_geometric.data.Batch, torch.Tensor]): The input data, which can either be
                a PyG Batch object or a raw Tensor.
            use_normalization (bool): Whether to normalize the input data.

        Returns:
            Union[torch_geometric.data.Batch, torch.Tensor]: The processed input, either as a Batch or a Tensor.
        """
        if isinstance(input, pyg.data.Batch):
            x = input.x
        elif isinstance(input, torch.Tensor):
            x = input
        else:
            raise TypeError(
                "Input must be either a torch_geometric.data.Batch or a torch.Tensor"
            )

        if use_normalization:
            x = x - self.node_x_mean.view(1, -1)
            x /= self.node_x_std.view(1, -1)

        x = self.encoder(x)

        if isinstance(input, pyg.data.Batch):
            input.x = x
            return input
        return x


@register_edge_encoder("VOCEdge")
class VOCEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        edge_x_mean = torch.tensor([0.07640745, 33.73478])
        edge_x_std = torch.tensor([0.0868775, 20.945076])
        self.register_buffer("edge_x_mean", edge_x_mean)
        self.register_buffer("edge_x_std", edge_x_std)

        if cfg.dataset.name in ["edge_wt_only_coord", "edge_wt_coord_feat"]:
            VOC_edge_input_dim = 1
        else:  # edge_wt_region_boundary by default
            VOC_edge_input_dim = 2
        self.encoder = torch.nn.Linear(VOC_edge_input_dim, emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(
        self, input: Union[pyg.data.Batch, torch.Tensor], use_normalization: bool = True
    ) -> Union[pyg.data.Batch, torch.Tensor]:
        """
        Forward method to handle either a Batch object or a Tensor object.

        Args:
            input (Union[torch_geometric.data.Batch, torch.Tensor]): The input data, which can either be
                a PyG Batch object (with edge_attr) or a raw Tensor representing edge attributes.
            use_normalization (bool): Whether to normalize the input data.

        Returns:
            Union[torch_geometric.data.Batch, torch.Tensor]: The processed input, either as a Batch or a Tensor.
        """
        if isinstance(input, pyg.data.Batch):
            edge_attr = input.edge_attr
        elif isinstance(input, torch.Tensor):
            edge_attr = input
        else:
            raise TypeError(
                "Input must be either a torch_geometric.data.Batch or a torch.Tensor"
            )

        if use_normalization:
            edge_attr = edge_attr - self.edge_x_mean.view(1, -1)
            edge_attr /= self.edge_x_std.view(1, -1)

        edge_attr = self.encoder(edge_attr)

        if isinstance(input, pyg.data.Batch):
            input.edge_attr = edge_attr
            return input
        return edge_attr


@register_node_encoder("COCONode")
class COCONodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        node_x_mean = torch.tensor(
            [
                4.6977347e-01,
                4.4679317e-01,
                4.0790915e-01,
                7.0808627e-02,
                6.8686441e-02,
                6.8498217e-02,
                6.7777938e-01,
                6.5244222e-01,
                6.2096798e-01,
                2.7554795e-01,
                2.5910738e-01,
                2.2901227e-01,
                2.4261935e02,
                2.8985367e02,
            ]
        )
        node_x_std = torch.tensor(
            [
                2.6218116e-01,
                2.5831082e-01,
                2.7416739e-01,
                5.7440419e-02,
                5.6832556e-02,
                5.7100497e-02,
                2.5929087e-01,
                2.6201612e-01,
                2.7675411e-01,
                2.5456995e-01,
                2.5140920e-01,
                2.6182330e-01,
                1.5152475e02,
                1.7630779e02,
            ]
        )

        self.register_buffer("node_x_mean", node_x_mean)
        self.register_buffer("node_x_std", node_x_std)
        self.encoder = torch.nn.Linear(VOC_node_input_dim, emb_dim)

    def forward(self, batch):
        x = batch.x - self.node_x_mean.view(1, -1)
        x /= self.node_x_std.view(1, -1)
        batch.x = self.encoder(x)
        return batch


@register_edge_encoder("COCOEdge")
class COCOEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        edge_x_mean = torch.tensor([0.07848548, 43.68736])
        edge_x_std = torch.tensor([0.08902349, 28.473562])
        self.register_buffer("edge_x_mean", edge_x_mean)
        self.register_buffer("edge_x_std", edge_x_std)
        VOC_edge_input_dim = 2 if cfg.dataset.name == "edge_wt_region_boundary" else 1
        self.encoder = torch.nn.Linear(VOC_edge_input_dim, emb_dim)

    def forward(self, batch):
        x = batch.edge_attr - self.edge_x_mean.view(1, -1)
        x /= self.edge_x_std.view(1, -1)
        batch.edge_attr = self.encoder(x)
        return batch
