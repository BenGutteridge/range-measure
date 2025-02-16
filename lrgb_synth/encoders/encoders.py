import torch
from torch import nn
import torch.nn.functional as F
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

from torch_geometric.graphgym.register import register_node_encoder

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()


# NEITHER OF THESE ARE USED - JUST FOR REFERENCE
# Actual encoders are imported from torch_geometric.graphgym.models.encoder
class AtomEncoder(nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x, pestat):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding


class DifferentiableAtomEncoder(nn.Module):
    """
    Alternative Atom encoder that takes pre-one-hot encoded float features 
    and applies a differentiable Linear layer rather than nn.Embedding.\
    """

    def __init__(self, emb_dim, minimal_sized_encodings=True):
        super(DifferentiableAtomEncoder, self).__init__()

        feature_dims = get_atom_feature_dims()
        if minimal_sized_encodings:
            # About 2/3 of feature channels for Atom encoder are unused, meaning a unnecessarily large input features.
            # This setting uses the minumum required.
            max_feat_dim_vals = [16, 2, 6, 6, 4, 0, 5, 1, 1]
        else:
            max_feat_dim_vals = [float("inf")] * 10
        feature_dims = [
            min(feat_dim, max_feat_dim_vals[i] + 1)
            for i, feat_dim in enumerate(feature_dims)
        ]
        self.cum_feature_dims = [
            sum(feature_dims[:i]) for i in range(len(feature_dims) + 1)
        ]

        self.linear_layers = nn.ModuleList()
        for dim in feature_dims:
            linear = nn.Linear(dim, emb_dim, bias=False)
            torch.nn.init.xavier_uniform_(linear.weight.data)
            self.linear_layers.append(linear)

    def forward(self, batch):
        # x: [num_nodes, SUM(feature_dims)] (concatenated one-hot encodings)

        encoded_features = 0
        for i, (lower_dim, upper_dim) in enumerate(
            zip(self.cum_feature_dims[:-1], self.cum_feature_dims[1:])
        ):
            # Get one-hot encoding of the i-th feature
            x = batch.x[:, lower_dim:upper_dim]

            encoded_features += self.linear_layers[i](x)  # [num_nodes, emb_dim]

        batch.x = encoded_features
        return batch


register_node_encoder("AtomDiff", DifferentiableAtomEncoder)
