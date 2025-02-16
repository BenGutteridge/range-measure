# This python library contains any functions which are used in creating the synthetic controllable tasks on graph datasets.

from typing import Optional
import math
from queue import Queue
import torch
import numpy as np
from torch_geometric.utils import (
    to_dense_adj,
    to_networkx,
    get_laplacian,
    dense_to_sparse,
)
import networkx as nx
from torch_geometric.data import Data
from typing import Callable
import matplotlib.pyplot as plt


# --------------------------|
#                          |
#    Distance Functions    |
#                          |
# --------------------------|


def get_distance_fn(dist_fn_name: str) -> Callable:
    dist_fn = DIST_FN_MAP.get(dist_fn_name)
    if dist_fn is None:
        raise ValueError(f"Unknown distance function: {dist_fn_name}")
    return dist_fn


def adjacency_matrix(data: Data):
    return to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze()


def gen_adjacency_powers(
    k: int,
    normalization: Optional[str] = "sym",
    self_loop: bool = False,
):

    def f(data):

        adj = to_dense_adj(
            data.edge_index, max_num_nodes=data.num_nodes
        ).squeeze()  # [N, N]
        if self_loop:
            adj = adj + torch.eye(adj.shape[0])
        deg = adj @ torch.ones(adj.shape[0])  # [N]]
        if normalization == "sym":
            D_inv_sqrt = torch.diag(
                deg.pow(-0.5).nan_to_num(posinf=0.0)
            )  # if isolated node set normalization to 0
            norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
            norm_adj = torch.linalg.matrix_power(norm_adj, k)
        elif normalization == "rw":
            D = torch.diag(
                deg.pow(-1).nan_to_num(posinf=0.0)
            )  # if isolated node set normalization to 0
            norm_adj = D @ adj
            norm_adj = torch.linalg.matrix_power(norm_adj, k)
        elif normalization is None:
            norm_adj = torch.linalg.matrix_power(adj, k)
        else:
            raise NotImplementedError
        return norm_adj

    return f


def gen_inverse_adjacency_powers(
    k: int, normalization: Optional[str] = "sym", self_loop: bool = False
):
    def f(data: Data):
        f = gen_adjacency_powers(k, normalization, self_loop)
        adj_power = f(data)
        inv_adj_power = adj_power.pow(-1)
        inv_adj_power = inv_adj_power.nan_to_num(posinf=0.0)
        return inv_adj_power

    return f


def gen_k_hop_mat(k: int) -> torch.Tensor:
    """
    Get dirac delta function for k-hop nodes

    Args:
        data (Data): Input data containing the graph information.
        k (int): The hop distance for which to compute the node pairs.

    Returns:
        torch.Tensor: A tensor representing the k-hop node pairs.
    """

    def f(data: Data):
        if hasattr(data, "edge_index_dense"):
            edge_index_dist = data.edge_index_dense
        else:
            edge_index_dist = dense_to_sparse(
                torch.ones([data.num_nodes, data.num_nodes])
            )[0]
        spd = torch.zeros([data.num_nodes, data.num_nodes])
        spd[edge_index_dist[0], edge_index_dist[1]] = data.edge_attr_spd.squeeze()
        spd_mat = (spd == k).float()
        D = torch.diag(spd_mat @ torch.ones(spd_mat.shape[0]))
        D_inv = torch.pow(D, -1).nan_to_num(posinf=0.0)
        spd_mat = D_inv @ spd_mat
        return spd_mat

    return f


def gen_k_nhbd_mat(k: int) -> torch.Tensor:
    """
    Rectangle distance_fn.

    Args:
        data (Data): Input data containing the graph information.
        k (int): The hop distance for which to compute the adjacency matrix.

    Returns:
        torch.Tensor: A tensor representing the k-hop adjacency matrix.
    """

    def f(data: Data):
        if hasattr(data, "edge_index_dense"):
            edge_index_dist = data.edge_index_dense
        else:
            edge_index_dist = dense_to_sparse(
                torch.ones([data.num_nodes, data.num_nodes])
            )[0]
        spd = torch.zeros([data.num_nodes, data.num_nodes])
        spd[edge_index_dist[0], edge_index_dist[1]] = data.edge_attr_spd.squeeze()
        spd_mat = (spd <= k).float()

        D = torch.diag(spd_mat @ torch.ones(spd_mat.shape[0]))
        D_inv = torch.pow(D, -1).nan_to_num(posinf=0.0)
        spd_mat = D_inv @ spd_mat

        return spd_mat

    return f


def inverse_shortest_path(data: Data) -> torch.Tensor:
    """
    Computes the inverse of the shortest path distance matrix for a given graph.

    Args:
        data (Data): Input data containing the graph information.

    Returns:
        torch.Tensor: A tensor representing the inverse of the shortest path distance matrix.
                      Infinite values are replaced with 0.0.
    """
    graph = to_networkx(data)
    spd_mat = nx.floyd_warshall_numpy(graph)
    spd_mat = torch.from_numpy(spd_mat)
    inv_spd_mat = torch.pow(spd_mat, -1.0)
    inv_spd_mat = torch.nan_to_num(inv_spd_mat, posinf=0.0)
    return inv_spd_mat


def shortest_path(data: Data) -> torch.Tensor:
    """
    Computes the shortest path distance matrix for a given graph.

    Args:
        data (Data): Input data containing the graph information.

    Returns:
        torch.Tensor: A tensor representing the inverse of the shortest path distance matrix.
                      Infinite values are replaced with 0.0.
    """
    graph = to_networkx(data)
    spd_mat = nx.floyd_warshall_numpy(graph)
    spd_mat = torch.from_numpy(spd_mat).float()
    spd = torch.nan_to_num(spd_mat, posinf=0.0)
    return spd


def inverse_exp_shortest_path(data: Data) -> torch.Tensor:
    """
    Computes the pairwise inverse exp(spd(i,j)) for all node pairs (i,j) in N x N.

    Parameters:
    - Data: an InMemoryDataset type of graph data

    Returns:
    - A 2D torch.Tensor of shape (n, n), where the (i, j)-th entry is 1 / exp(spd(i,j)).
    """
    graph = to_networkx(data)
    spd_mat = nx.floyd_warshall_numpy(graph)
    spd_mat = torch.from_numpy(spd_mat)
    exp_spd_mat = torch.exp(spd_mat)
    inv_exp_spd_mat = torch.pow(exp_spd_mat, -1.0)

    return inv_exp_spd_mat


def resistance_distance(data: Data) -> torch.Tensor:
    """
    Computes the pairwise resistance distance.

    Parameters:
    - Data: an InMemoryDataset type of graph data

    Returns:
    - A 2D torch.Tensor of shape (n, n), where the (i, j)-th entry is RD(i,j)).
    """
    graph = to_networkx(data, to_undirected=True)
    res_dist = resistance_distance_disconnected(graph)
    return res_dist


def resistance_distance_disconnected(G: nx.classes.graph.Graph) -> torch.Tensor:
    # Initialize resistance distance matrix
    nodes = list(G.nodes)
    n = len(nodes)
    resistance_matrix = torch.zeros((n, n))  # Fill with 0 by default

    # Check if the graph is connected
    if nx.is_connected(G):
        res_dist = nx.resistance_distance(G)
        for i, u in enumerate(G.nodes):
            for j, v in enumerate(G.nodes):
                resistance_matrix[nodes.index(u), nodes.index(v)] = res_dist[u][v]

    else:
        # Get the connected components
        components = list(nx.connected_components(G))

        # Create a mapping from node to component index
        node_to_component = {}
        for idx, component in enumerate(components):
            for node in component:
                node_to_component[node] = idx

        # Fill the resistance distances within each component
        for component in components:
            subgraph = G.subgraph(component)
            sub_resistance = nx.resistance_distance(subgraph)
            for i, u in enumerate(component):
                for j, v in enumerate(component):
                    resistance_matrix[nodes.index(u), nodes.index(v)] = sub_resistance[
                        u
                    ][v]

    return resistance_matrix


def inverse_resistance_distance(data: Data) -> torch.Tensor:
    """
    Computes the pairwise inverse exp(spd(i,j)) for all node pairs (i,j) in N x N.

    Parameters:
    - Data: an InMemoryDataset type of graph data

    Returns:
    - A 2D torch.Tensor of shape (n, n), where the (i, j)-th entry is 1 / RD(i,j)).
    """
    res_dist = resistance_distance(data)
    inv_rd_mat = torch.pow(res_dist, -1)
    return inv_rd_mat


def sqrt_adjacency_dist_fn(f: Callable) -> Callable:
    """
    Wrapper for adjacency-based distance functions that
    square roots adjacency powers (to give greater relative weight to distant node pairs)
    """

    def wrapper(data: Data):
        mat = f(data)
        mat = mat.pow(0.5)
        D = torch.diag(mat @ torch.ones(mat.shape[1]))
        D_inv_sqrt = D.pow(-0.5).nan_to_num(posinf=0.0)
        mat = D_inv_sqrt @ mat @ D_inv_sqrt  # to normalize to ensure bounded spectrum
        return mat

    return wrapper


DIST_FN_MAP = {
    "adjacency_matrix": adjacency_matrix,
    "inv_sp": inverse_shortest_path,
    "inv_exp_sp": inverse_exp_shortest_path,
    "inv_resistance": inverse_resistance_distance,
    "resistance": resistance_distance,
    "spd": shortest_path,
}

# Add adjacency power entries with and without self loops
# And *inverse* adjacency power entries with and without self loops
normalizations = ["sym", "rw", None]
max_k = 10
for k in range(1, max_k + 1):
    for normalization in normalizations:
        # Adj
        DIST_FN_MAP[f"adjacency_power_{normalization}_{k}"] = gen_adjacency_powers(
            k, normalization
        )
        # Adj + self loop
        DIST_FN_MAP[f"adjacency_self_loop_power_{normalization}_{k}"] = (
            gen_adjacency_powers(k, normalization, self_loop=True)
        )
        # Add inverse adjacency power entries
        DIST_FN_MAP[f"inv_adjacency_power_{normalization}_{k}"] = (
            gen_inverse_adjacency_powers(k, normalization)
        )
        # Inverse adj + self loop
        DIST_FN_MAP[f"inv_adjacency_self_loop_power_{normalization}_{k}"] = (
            gen_inverse_adjacency_powers(k, normalization, self_loop=True)
        )
    # add the hop/dirac task
    DIST_FN_MAP[f"hop_{k}"] = gen_k_hop_mat(k)
    # add the nbhd/rectangle task
    DIST_FN_MAP[f"nbhd_{k}"] = gen_k_nhbd_mat(k)

    # add the sqrt of adjacency power distance functions
    normalization = "sym"
    DIST_FN_MAP[f"adjacency_self_loop_power_{normalization}_{k}_sqrt"] = (
        sqrt_adjacency_dist_fn(gen_adjacency_powers(k, normalization, self_loop=True))
    )
    DIST_FN_MAP[f"inv_adjacency_self_loop_power_{normalization}_{k}_sqrt"] = (
        sqrt_adjacency_dist_fn(
            gen_inverse_adjacency_powers(k, normalization, self_loop=True)
        )
    )


# --------------------------|
#                          |
#   Interaction Functions  |
#                          |
# --------------------------|


def get_interaction_fn(interaction_fn_name: str) -> Callable:
    interaction_fn = INTERACTION_FN_MAP.get(interaction_fn_name)
    if interaction_fn is None:
        raise ValueError(f"Unknown interaction function: {interaction_fn_name}")
    return interaction_fn


def L2_norm(X: torch.Tensor) -> torch.Tensor:
    """
    Computes the L2 norm (Euclidean distance) between each pair of rows in the input tensor.
    Args:
        X (torch.Tensor): A 2D tensor of shape (n, d) where n is the number of samples and d is the number of features.
    Returns:
        torch.Tensor: A 2D tensor of shape (n, n) containing the pairwise L2 norms between rows of the input tensor.
    """
    diff = X.unsqueeze(1) - X.unsqueeze(0)  # [n, n, d]
    # Squaring ensures nonzero Hessian in 1dim case
    norm = torch.linalg.norm(diff, dim=2).pow(2)  # [n, n]
    return norm


def cosine_similarity(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the pairwise cosine similarity matrix for the rows of X.
    Parameters:
    - X: A 2D torch.Tensor of shape (n, d), where n is the number of vectors,
      and d is the dimensionality of each vector.
    Returns:
    - A 2D torch.Tensor of shape (n, n), where the (i, j)-th entry is the cosine similarity between X[i] and X[j].
    """
    X_normalized = torch.nn.functional.normalize(
        X, p=2, dim=1
    )  # Normalize rows to unit vectors
    cosine_sim_matrix = X_normalized @ X_normalized.T
    return cosine_sim_matrix  # squeeze


def opp_lin(X: torch.Tensor) -> torch.Tensor:
    """interaction function is (X_s, X_t) -> X_s, just selects the sources features and copies it"""
    X = X.unsqueeze(1)
    X = X.repeat([1, X.shape[0], X.shape[1]])  # NxNxd, but I think d=1
    return X.squeeze()


INTERACTION_FN_MAP = {
    "L2_norm": L2_norm,
    "cosine_similarity": cosine_similarity,
    "opp_lin": opp_lin,
}
