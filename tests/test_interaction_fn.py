import unittest
import torch
from torch import nn
from longrange.syn_task_utils import get_interaction_fn, L2_norm, cosine_similarity


class TestInteractionFunctions(unittest.TestCase):
    def setUp(self):
        """
        Set up mock data for testing.
        """
        # Example tensor with 3 data points, each with 2 features
        self.X = torch.tensor([[1.87, 2.21], [3.21, 4.58], [5.1, 6.0]])

    def test_get_interaction_fn_valid(self):
        """
        Test if get_interaction_fn correctly retrieves functions from INTERACTION_FN_MAP.
        """
        l2_fn = get_interaction_fn("L2_norm")
        cos_fn = get_interaction_fn("Cosine Similarity")

        self.assertEqual(l2_fn, L2_norm)
        self.assertEqual(cos_fn, cosine_similarity)

    def test_get_interaction_fn_invalid(self):
        """
        Test if get_interaction_fn raises an error for invalid function names.
        """
        with self.assertRaises(ValueError):
            get_interaction_fn("NonExistentFunction")

    def test_L2_norm(self):
        """
        Test L2_norm function to ensure correct pairwise L2 norms are calculated.
        """
        l2_norm_matrix = L2_norm(self.X)

        # Calculate expected L2 norm matrix manually
        expected_matrix = torch.tensor(
            [
                [0.0000, 2.8284, 5.6569],
                [2.8284, 0.0000, 2.8284],
                [5.6569, 2.8284, 0.0000],
            ]
        )

        self.assertTrue(torch.allclose(l2_norm_matrix, expected_matrix, atol=1e-4))

    def test_cosine_similarity(self):
        """
        Test cosine_similarity function to ensure it returns expected pairwise cosine similarities.
        """
        cos_sim_matrix = cosine_similarity(self.X)

        # Expected cosine similarity matrix
        cos = nn.CosineSimilarity(dim=1)

        expected_matrix = cos(self.X.unsqueeze(1), self.X.unsqueeze(0))
        print(expected_matrix)
        print(cos_sim_matrix)

        self.assertTrue(torch.allclose(cos_sim_matrix, expected_matrix, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
