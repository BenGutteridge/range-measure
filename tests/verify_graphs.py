"""For verifying that synthetically generated graphs look as they should during development."""

import torch
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path


def plot_graph(edge_index: torch.Tensor, save_path: str = "plots") -> None:
    """
    Plots a graph from a PyTorch edge index tensor using NetworkX and saves it as a PNG.

    Args:
        edge_index (torch.Tensor): A 2x|E| edge index tensor.
        save_path (str): The folder path where the graph PNG should be saved.
    """
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape 2x|E|")

    graph = nx.Graph()
    graph.add_edges_from(edge_index.t().tolist())

    pos = nx.spring_layout(graph)

    plt.figure(figsize=(8, 8))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=500,
        font_size=10,
    )
    plt.title("Graph Visualization")

    save_folder = Path(save_path)
    save_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_folder / "graph_verification.png")
    plt.close()
    print(f"Graph saved as {save_folder / 'graph_verification.png'}")
