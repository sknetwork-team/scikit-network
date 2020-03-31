#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for format.py"""

import unittest

from sknetwork.data import StarWars, House, CycleDirected
from sknetwork.utils.format import *
from sknetwork.utils.check import is_symmetric


# noinspection PyMissingOrEmptyDocstring
class TestFormats(unittest.TestCase):

    def setUp(self):
        self.biadjacency = StarWars().biadjacency

    def test_dir2undir(self):
        adjacency = CycleDirected(3).adjacency
        undirected_graph = directed2undirected(adjacency)
        self.assertEqual(undirected_graph.shape, adjacency.shape)
        self.assertTrue(is_symmetric(undirected_graph))

        adjacency = House().adjacency
        error = 0.5 * directed2undirected(adjacency) - adjacency
        self.assertEqual(error.nnz, 0)

    def test_bip2dir(self):
        n_nodes = self.biadjacency.shape[0] + self.biadjacency.shape[1]
        directed_graph = bipartite2directed(self.biadjacency)
        self.assertEqual(directed_graph.shape, (n_nodes, n_nodes))

        n1, n2 = self.biadjacency.shape
        slr = SparseLR(self.biadjacency, [(np.ones(n1), np.ones(n2))])
        directed_graph = bipartite2directed(slr)
        self.assertTrue(type(directed_graph) == SparseLR)

    def test_bip2undir(self):
        n_nodes = self.biadjacency.shape[0] + self.biadjacency.shape[1]
        undirected_graph = bipartite2undirected(self.biadjacency)
        self.assertEqual(undirected_graph.shape, (n_nodes, n_nodes))
        self.assertTrue(is_symmetric(undirected_graph))

        n1, n2 = self.biadjacency.shape
        slr = SparseLR(self.biadjacency, [(np.ones(n1), np.ones(n2))])
        undirected_graph = bipartite2undirected(slr)
        self.assertTrue(type(undirected_graph) == SparseLR)
