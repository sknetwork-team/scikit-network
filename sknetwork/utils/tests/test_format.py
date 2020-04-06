#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for format.py"""

import unittest

from sknetwork.data import star_wars, house, cyclic_digraph
from sknetwork.utils.format import *
from sknetwork.utils.check import is_symmetric


class TestFormats(unittest.TestCase):

    def setUp(self):
        """Basic biadjacency for tests."""
        self.biadjacency = star_wars()

    def test_dir2undir(self):
        adjacency = cyclic_digraph(3)
        undirected_graph = directed2undirected(adjacency)
        self.assertEqual(undirected_graph.shape, adjacency.shape)
        self.assertTrue(is_symmetric(undirected_graph))

        adjacency = house()
        error = 0.5 * directed2undirected(adjacency) - adjacency
        self.assertEqual(error.nnz, 0)

    def test_bip2dir(self):
        n_row, n_col = self.biadjacency.shape
        n = n_row + n_col

        directed_graph = bipartite2directed(self.biadjacency)
        self.assertEqual(directed_graph.shape, (n, n))

        slr = SparseLR(self.biadjacency, [(np.ones(n_row), np.ones(n_col))])
        directed_graph = bipartite2directed(slr)
        self.assertTrue(type(directed_graph) == SparseLR)

    def test_bip2undir(self):
        n_row, n_col = self.biadjacency.shape
        n = n_row + n_col

        undirected_graph = bipartite2undirected(self.biadjacency)
        self.assertEqual(undirected_graph.shape, (n, n))
        self.assertTrue(is_symmetric(undirected_graph))

        slr = SparseLR(self.biadjacency, [(np.ones(n_row), np.ones(n_col))])
        undirected_graph = bipartite2undirected(slr)
        self.assertTrue(type(undirected_graph) == SparseLR)
