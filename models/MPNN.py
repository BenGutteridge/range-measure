import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch.nn import Module, Dropout, LayerNorm, Identity
import torch.nn.functional as F
from typing import Tuple
import numpy as np
from typing import Optional

from helpers.classes import GNNArgs, Pool


class MPNN(Module):
    def __init__(
        self,
        gnn_args: GNNArgs,
        pool: Pool,
    ):
        super(MPNN, self).__init__()
        self.gnn_args = gnn_args

        self.num_layers = gnn_args.num_layers
        self.net = gnn_args.load_net()
        self.use_encoders = gnn_args.dataset_encoders.use_encoders()

        layer_norm_cls = LayerNorm if gnn_args.layer_norm else Identity
        self.hidden_layer_norm = layer_norm_cls(gnn_args.hidden_dim)
        self.skip = gnn_args.skip
        self.dropout = Dropout(p=gnn_args.dropout)
        self.act = gnn_args.act_type.get()

        # Encoder types
        self.dataset_encoder = gnn_args.dataset_encoders
        self.bond_encoder = self.dataset_encoder.edge_encoder(
            emb_dim=gnn_args.hidden_dim, model_type=gnn_args.model_type
        )

        # Pooling function to generate whole-graph embeddings
        self.pooling = pool.get()

        # Indicates if the model is run without a head/decoder
        self._headless = False

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        pestat: Optional[list[Tensor]],
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
    ) -> Tuple[Tensor, Tensor]:
        result = 0
        """"""
        # Bond encode
        if edge_attr is None or self.bond_encoder is None:
            edge_embedding = None
        else:
            edge_embedding = self.bond_encoder(edge_attr)

        # Node encode
        x = self.net[0](x, pestat)  # (N, F) encoder
        if not self.use_encoders:
            x = self.dropout(x)
            x = self.act(x)

        for gnn_idx in range(self.num_layers):  # num layers repeated in net
            x = self.hidden_layer_norm(x)

            out = self.net[1 + gnn_idx](
                x=x,
                edge_index=edge_index,
                edge_attr=edge_embedding,
            )
            out = self.dropout(out)
            out = self.act(out)

            if self.skip:
                x = x + out
            else:
                x = out

        x = self.hidden_layer_norm(x)
        if (
            not self._headless
        ):  # if headless, skip pooling and decoder; leave feats at node level
            x = self.pooling(x, batch=batch)
            x = self.net[-1](x)  # decoder
        result = result + x

        return result

    def create_edge_weight(
        self, edge_index: Adj, keep_in_prob: Tensor, keep_out_prob: Tensor
    ) -> Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        return edge_in_prob * edge_out_prob

    def make_headless(self, undo: bool = False):
        if undo:
            assert self._headless, "Model is not headless"
            self._headless = False
        else:
            assert not self._headless, "Model is already headless"
            self._headless = True
