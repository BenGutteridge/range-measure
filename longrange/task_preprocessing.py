from typing import Optional
import torch
from torch import Tensor
import torch_geometric
from typing import Callable
from torch_geometric.data import Data
from longrange.syn_task_utils import get_distance_fn, get_interaction_fn
from loguru import logger
from torch.autograd.functional import jacobian
from torch.func import hessian
from torch_geometric.utils.sparse import dense_to_sparse


def shuffle(tensor):
    idx = torch.randperm(len(tensor))
    return tensor[idx]


def task_specific_preprocessing(data, cfg):
    """Task-specific preprocessing before the dataset is logged and finalized.

    Args:
        data: PyG graph
        cfg: Main configuration node

    Returns:
        Extended PyG Data object.
    """
    if cfg.gnn.head == "infer_links":
        N = data.x.size(0)
        idx = torch.arange(N, dtype=torch.long)
        complete_index = torch.stack([idx.repeat_interleave(N), idx.repeat(N)], 0)

        data.edge_attr = None

        if cfg.dataset.infer_link_label == "edge":
            labels = torch.empty(N, N, dtype=torch.long)
            non_edge_index = (
                (complete_index.T.unsqueeze(1) != data.edge_index.T)
                .any(2)
                .all(1)
                .nonzero()[:, 0]
            )
            non_edge_index = shuffle(non_edge_index)[: data.edge_index.size(1)]
            edge_index = (
                (complete_index.T.unsqueeze(1) == data.edge_index.T)
                .all(2)
                .any(1)
                .nonzero()[:, 0]
            )

            final_index = shuffle(torch.cat([edge_index, non_edge_index]))
            data.complete_edge_index = complete_index[:, final_index]

            labels.fill_(0)
            labels[data.edge_index[0], data.edge_index[1]] = 1

            assert labels.flatten()[final_index].mean(dtype=torch.float) == 0.5
        else:
            raise ValueError(
                f"Infer-link task {cfg.dataset.infer_link_label} not available."
            )

        data.y = labels.flatten()[final_index]

    supported_encoding_available = (
        cfg.posenc_LapPE.enable
        or cfg.posenc_RWSE.enable
        or cfg.posenc_GraphormerBias.enable
    )

    if cfg.dataset.name == "TRIANGLES":

        # If encodings are present they can append to the empty data.x
        if not supported_encoding_available:
            data.x = torch.zeros((data.x.size(0), 1))
        data.y = data.y.sub(1).to(torch.long)

    if cfg.dataset.name == "CSL":

        # If encodings are present they can append to the empty data.x
        if not supported_encoding_available:
            data.x = torch.zeros((data.num_nodes, 1))
        else:
            data.x = torch.zeros((data.num_nodes, 0))

    return data


def syn_task_preprocessing(
    data: Data,
    distance_fn: str,
    interaction_fn: str,
    feature_alpha: float,
    scalar_feats: bool = True,
    track_range: bool = False,
    graph_level: bool = False,
    use_jacobian: bool = False,
) -> Data:
    """Task-specific preprocessing before the dataset is logged and finalized.

    Args:
        data: PyG graph
        distance_fn: name of distance function, e.g. adjacency, shortest path
        interaction_fn: name of interaction function, e.g. L2
        feature_alpha: node features will be alpha * data.x + (1-alpha) * uniform random
        scalar_feats: if True, node features are purely synthetic, 1D
        track_range: if True, compute and store the Jacobian of the task function
        graph_level: indicates if the task is graph_level or not
        use_jacobian: indicates if the final_layer jacobian shoud be computed instead of the hessian in the graph-level case

    Returns:
        Edited PyG Data object with (partially) synthetic node feats and label(s).
    """
    data.isolated_nodes = False  # flag for isolated nodes
    if torch_geometric.utils.contains_isolated_nodes(data.edge_index, data.num_nodes):

        _, _, mask = torch_geometric.utils.remove_isolated_nodes(
            data.edge_index, num_nodes=data.num_nodes
        )
        logger.warning(f"{(~mask).sum()} out of {len(mask)} isolated nodes detected.")
        data.isolated_nodes = True
    dist_fn = get_distance_fn(distance_fn)
    interaction_fn = get_interaction_fn(interaction_fn)

    if scalar_feats:  # purely synthetic, 1D node features
        assert feature_alpha == 0.0, "Scalar features are only supported for alpha=0."
        data.x = torch.randn(data.num_nodes, 1, requires_grad=True)  # |V| x 1
    else:  # use a mix of original and synthetic features with original dimensionality
        data.x = feature_alpha * data.x + (1 - feature_alpha) * torch.rand(data.x.shape)

    def gen_node_label(x: Tensor) -> Tensor:
        """Differentiable function generating the labels, so we can compute its Jacobian"""
        dist_int = dist_fn(data) * interaction_fn(x)  # |V| x |V|
        return dist_int.sum(axis=0).unsqueeze(1).float()  # |V| x 1

    def gen_graph_label(x: Tensor) -> Tensor:
        """Differentiable function generating the labels to compute the Hessian"""
        node_level = gen_node_label(x)
        return node_level.mean(axis=0).unsqueeze(0)  # [1, 1]

    if graph_level:
        data.y = gen_graph_label(data.x)
    else:
        data.y = gen_node_label(data.x)

    edge_index_dense = dense_to_sparse(
        torch.ones([data.num_nodes, data.num_nodes]),
    )[0]

    if track_range:
        if graph_level and not use_jacobian:
            task_hessian = hessian(gen_graph_label)(data.x).squeeze(
                0
            )  # [d_out, N, d_in, N, d_in], assumes batch_size = 1
            edge_attr_hes = task_hessian[
                :, edge_index_dense[0], :, edge_index_dense[1], :
            ]  # [N**2, d_out, d_in, d_in]
            assert torch.allclose(  # approximates torch.equal()
                edge_attr_hes.reshape(
                    [
                        data.num_nodes,
                        data.num_nodes,
                        task_hessian.shape[0],
                        task_hessian.shape[2],
                        task_hessian.shape[4],
                    ]
                ).permute(2, 0, 3, 1, 4),
                task_hessian,
            )  # check that NxN Jacobian matrix has been reshaped into N**2 vector correctly
            data.edge_attr_hes = edge_attr_hes  # [N**2, d_out, d_in, d_in]
        else:
            task_jacobian = jacobian(
                gen_node_label, data.x
            )  # [N, dim_out, N, dim_in]  # This might be inefficient but it generallizes to all dist/interaction functions
            edge_attr_jac = task_jacobian[
                edge_index_dense[0], :, edge_index_dense[1], :
            ]  # [N**2, d_out, d_in]
            assert torch.allclose(  # approximates torch.equal()
                edge_attr_jac.reshape(
                    [
                        data.num_nodes,
                        data.num_nodes,
                        task_jacobian.shape[1],
                        task_jacobian.shape[3],
                    ]
                ).permute(0, 2, 1, 3),
                task_jacobian,
            )  # check that NxN Jacobian matrix has been reshaped into N**2 vector correctly
            data.edge_attr_jac = edge_attr_jac.reshape(
                [edge_attr_jac.shape[0], edge_attr_jac.shape[1], -1]
            )  # flatten two last dimensions to store the jacobian matrix as a vector
    return data


def syn_topology_preprocessing(
    data: Data,
) -> Data:
    """Topology-specific preprocessing before the dataset is logged and finalized.

    Args:
        data: PyG graph

    Returns:
        Edited PyG Data object with resistance distance and shortest path distances.
    """
    data = add_distance_attrs(data, "spd")
    data = add_distance_attrs(data, "resistance")

    return data


def add_distance_attrs(data: Data, distance_fn: str) -> Data:
    """Compute and store node-to-node distance attributes in a PyG Data object.

    Args:
        data: PyG graph
        distance_fn: name of distance function, 'spd' or 'res'

    Returns:
        Extended PyG Data object.
    """
    edge_index_dense = dense_to_sparse(
        torch.ones([data.num_nodes, data.num_nodes]),
    )[0]

    dist_fct = get_distance_fn(distance_fn)
    dist = dist_fct(data)  # [N, N]
    edge_attr = dist[edge_index_dense[0], edge_index_dense[1]]

    if distance_fn == "spd":
        data.edge_attr_spd = edge_attr
    elif distance_fn == "resistance":
        data.edge_attr_res = edge_attr
    else:
        raise ValueError(f"Unknown distance function {distance_fn}")

    data.edge_index_dense = edge_index_dense
    return data
