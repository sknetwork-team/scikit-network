"""Tests for shortest paths with explicitly bipartite input."""
import unittest

import numpy as np
from scipy import sparse

from sknetwork.path import get_shortest_path


class TestShortestPathBipartite(unittest.TestCase):

    def test_square(self):
        biadjacency = sparse.csr_matrix([[1, 1], [0, 1]])
        cases = [
            ({'source': 0}, {(0, 2), (0, 3), (3, 1)}),
            ({'source': [0]}, {(0, 2), (0, 3), (3, 1)}),
            ({'source_row': 0}, {(0, 2), (0, 3), (3, 1)}),
            ({'source_col': 0}, {(2, 0), (0, 3), (3, 1)}),
            ({'source': np.array([0, 1])}, {(0, 2), (0, 3), (1, 3)}),
            ({'source_row': 1, 'source_col': 0}, {(1, 3), (2, 0)}),
        ]
        for sources, edges in cases:
            with self.subTest(sources=sources):
                path = get_shortest_path(
                    biadjacency, force_bipartite=True, **sources)
                self.assertEqual(path.shape, (4, 4))
                self.assertEqual(set(zip(*path.nonzero())), edges)
        np.testing.assert_array_equal(biadjacency.toarray(), [[1, 1], [0, 1]])

    def test_rectangular(self):
        biadjacency = sparse.csr_matrix([[1, 1, 0], [0, 1, 1]])
        cases = [
            ({'source_row': 1}, {(1, 3), (1, 4), (3, 0), (0, 2)}),
            ({'source_col': 2}, {(4, 1), (1, 3), (3, 0), (0, 2)}),
            ({'source': [0, 1]}, {(0, 2), (0, 3), (1, 3), (1, 4)}),
        ]
        for sources, edges in cases:
            for force_bipartite in (False, True):
                with self.subTest(sources=sources, force=force_bipartite):
                    path = get_shortest_path(
                        biadjacency, force_bipartite=force_bipartite,
                        **sources)
                    self.assertEqual(path.shape, (5, 5))
                    self.assertEqual(set(zip(*path.nonzero())), edges)

    def test_empty_single_edge_and_isolated_nodes(self):
        cases = [
            ([[0, 0], [0, 0]], set()),
            ([[1]], {(0, 1)}),
            ([[1, 0], [0, 0]], {(0, 2)}),
        ]
        for matrix, edges in cases:
            with self.subTest(matrix=matrix):
                biadjacency = sparse.csr_matrix(matrix)
                path = get_shortest_path(
                    biadjacency, source=0, force_bipartite=True)
                n_nodes = sum(biadjacency.shape)
                self.assertEqual(path.shape, (n_nodes, n_nodes))
                self.assertEqual(set(zip(*path.nonzero())), edges)

    def test_square_default(self):
        adjacency = sparse.csr_matrix([[1, 1], [0, 1]])
        path = get_shortest_path(adjacency, source=0)
        self.assertEqual(path.shape, (2, 2))
        self.assertEqual(set(zip(*path.nonzero())), {(0, 1)})
