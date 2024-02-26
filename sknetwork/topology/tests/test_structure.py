#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for structure.py"""
import unittest

import numpy as np
from scipy import sparse

from sknetwork.data import star_wars, house, cyclic_digraph, cyclic_graph, linear_digraph, linear_graph
from sknetwork.topology import get_connected_components, get_largest_connected_component
from sknetwork.topology import is_connected, is_bipartite
from sknetwork.utils.format import bipartite2undirected, directed2undirected


class TestStructure(unittest.TestCase):

    def test_cc(self):
        adjacency = linear_digraph(3)
        labels = get_connected_components(adjacency, connection='weak')
        self.assertEqual(len(set(labels)), 1)
        adjacency = linear_digraph(3)
        labels = get_connected_components(adjacency, connection='weak', force_bipartite=True)
        self.assertEqual(len(set(labels)), 4)
        biadjacency = sparse.csr_matrix(([1, 1, 1, 1], [[1, 2, 3, 3], [1, 2, 2, 3]]))
        labels = get_connected_components(biadjacency)
        self.assertEqual(len(set(labels)), 3)

    def test_connected(self):
        adjacency = cyclic_digraph(3)
        self.assertEqual(is_connected(adjacency), True)
        adjacency = linear_digraph(3)
        self.assertEqual(is_connected(adjacency, connection='strong'), False)
        biadjacency = star_wars()
        self.assertEqual(is_connected(biadjacency), True)

    def test_largest_cc(self):
        adjacency = cyclic_digraph(3)
        adjacency += adjacency.T
        largest_cc, index = get_largest_connected_component(adjacency, return_index=True)

        self.assertAlmostEqual(np.linalg.norm(largest_cc.toarray() - np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])), 0)
        self.assertEqual(np.linalg.norm(index - np.array([0, 1, 2])), 0)

        biadjacency = star_wars(metadata=False)
        largest_cc, index = get_largest_connected_component(biadjacency, return_index=True)
        self.assertAlmostEqual(np.linalg.norm(largest_cc.toarray() - np.array([[1, 0, 1],
                                                                               [1, 0, 0],
                                                                               [1, 1, 1],
                                                                               [0, 1, 1]])), 0)
        self.assertEqual(np.linalg.norm(index - np.array([0, 1, 2, 3, 0, 1, 2])), 0)

        biadjacency = sparse.csr_matrix(([1, 1, 1, 1], [[1, 2, 3, 3], [1, 2, 2, 3]]))
        largest_cc, index = get_largest_connected_component(biadjacency, force_bipartite=True, return_index=True)
        self.assertEqual(largest_cc.shape[0], 2)
        self.assertEqual(len(index), 4)

        self.assertTrue(isinstance(get_largest_connected_component(adjacency, return_index=False), sparse.csr_matrix))

    def test_is_bipartite(self):
        biadjacency = star_wars(metadata=False)
        adjacency = bipartite2undirected(biadjacency)
        self.assertTrue(is_bipartite(adjacency))

        bipartite, biadjacency_pred, _, _ = is_bipartite(adjacency, return_biadjacency=True)
        self.assertEqual(bipartite, True)
        self.assertEqual(np.all(biadjacency.data == biadjacency_pred.data), True)

        adjacency = sparse.identity(2, format='csr')
        bipartite, biadjacency, _, _ = is_bipartite(adjacency, return_biadjacency=True)
        self.assertEqual(bipartite, False)
        self.assertIsNone(biadjacency)

        adjacency = directed2undirected(cyclic_digraph(3))
        bipartite, biadjacency, _, _ = is_bipartite(adjacency, return_biadjacency=True)
        self.assertEqual(bipartite, False)
        self.assertIsNone(biadjacency)

        with self.assertRaises(ValueError):
            is_bipartite(cyclic_digraph(3))

        self.assertFalse(is_bipartite(sparse.eye(3)))

        adjacency = directed2undirected(cyclic_digraph(3))
        bipartite = is_bipartite(adjacency, return_biadjacency=False)
        self.assertEqual(bipartite, False)
