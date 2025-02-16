import unittest
import torch
from torch_geometric.data import Data
import networkx as nx
from longrange.syn_task_utils import (
    get_distance_fn,
    adjacency_matrix,
    inverse_shortest_path,
    inverse_resistance_distance,
)


class TestDistanceFunctions(unittest.TestCase):
    def setUp(self):
        """
        Set up mock data for testing.
        """
        # Example edge_index for a graph with 3 nodes: 0 -> 1, 1 -> 2
        self.edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        self.shortest_path = torch.tensor([])
        self.num_nodes = 3
        self.data = Data(edge_index=self.edge_index, num_nodes=self.num_nodes)

    def test_get_distance_fn_valid(self):
        """
        Test if get_distance_fn correctly retrieves functions from DIST_FN_MAP.
        """
        adj_fn = get_distance_fn("adjacency_matrix")
        isp_fn = get_distance_fn("Inverse Shortest Path")
        ird_fn = get_distance_fn("Inverse Resistance Distance")

        self.assertEqual(adj_fn, adjacency_matrix)
        self.assertEqual(isp_fn, inverse_shortest_path)
        self.assertEqual(ird_fn, inverse_resistance_distance)

    def test_get_distance_fn_invalid(self):
        """
        Test if get_distance_fn raises an error for invalid function names.
        """
        with self.assertRaises(ValueError):
            get_distance_fn("NonExistentFunction")

    def test_adjacency_matrix(self):
        """
        Test adjacency_matrix function to ensure correct adjacency matrix output.
        """
        adj = adjacency_matrix(self.data)
        expected_adj = torch.tensor(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float
        )

        self.assertTrue(torch.equal(adj, expected_adj))

    def test_inverse_shortest_path(self):
        """
        Test inverse_shortest_path function to ensure correct inverse shortest path output.
        """
        inv_spd = inverse_shortest_path(self.data)

        # Calculate expected shortest path distance matrix using networkx
        graph = nx.Graph()
        graph.add_edges_from(self.edge_index.t().tolist())
        spd_mat = nx.floyd_warshall_numpy(graph)
        expected_inv_spd = torch.pow(torch.tensor(spd_mat), -1.0)

        self.assertTrue(torch.allclose(inv_spd, expected_inv_spd, atol=1e-6))

    def test_inverse_resistance_dist(self):
        inv_rd = inverse_resistance_distance(self.data)


if __name__ == "__main__":
    unittest.main()
