#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for structure.py"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.basics.structure import largest_connected_component, is_bipartite
from sknetwork.data import star_wars, cyclic_digraph
from sknetwork.utils.format import bipartite2undirected, directed2undirected


class TestStructure(unittest.TestCase):

    def test_largest_cc(self):
        adjacency = cyclic_digraph(3)
        adjacency += adjacency.T
        largest_cc, indices = largest_connected_component(adjacency, return_labels=True)

        self.assertAlmostEqual(np.linalg.norm(largest_cc.toarray() - np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])), 0)
        self.assertEqual(np.linalg.norm(indices - np.array([0, 1, 2])), 0)

        biadjacency = star_wars(metadata=False)
        largest_cc, indices = largest_connected_component(biadjacency, return_labels=True)

        self.assertAlmostEqual(np.linalg.norm(largest_cc.toarray() - np.array([[1, 0, 1],
                                                                               [1, 0, 0],
                                                                               [1, 1, 1],
                                                                               [0, 1, 1]])), 0)
        self.assertEqual(np.linalg.norm(indices[0] - np.array([0, 1, 2, 3])), 0)
        self.assertEqual(np.linalg.norm(indices[1] - np.array([0, 1, 2])), 0)

        self.assertTrue(isinstance(largest_connected_component(adjacency, return_labels=False), sparse.csr_matrix))

    def test_is_bipartite(self):
        biadjacency = star_wars(metadata=False)
        adjacency = bipartite2undirected(biadjacency)
        self.assertTrue(is_bipartite(adjacency))

        bipartite, pred = is_bipartite(adjacency, return_biadjacency=True)
        self.assertEqual(bipartite, True)
        self.assertEqual(np.all(biadjacency.data == pred.data), True)

        adjacency = sparse.identity(2, format='csr')
        bipartite, biadjacency = is_bipartite(adjacency, return_biadjacency=True)
        self.assertEqual(bipartite, False)
        self.assertIsNone(biadjacency)

        adjacency = directed2undirected(cyclic_digraph(3))
        bipartite, biadjacency = is_bipartite(adjacency, return_biadjacency=True)
        self.assertEqual(bipartite, False)
        self.assertIsNone(biadjacency)

        with self.assertRaises(ValueError):
            is_bipartite(cyclic_digraph(3))

        self.assertTrue(~is_bipartite(sparse.eye(3)))
