#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for adjacency_formats.py"""


import unittest
from sknetwork.toy_graphs import star_wars_villains, rock_paper_scissors, house
from sknetwork.utils.adjacency_formats import *
from sknetwork.utils.checks import is_symmetric


class TestFormats(unittest.TestCase):

    def setUp(self):
        self.adjacency = rock_paper_scissors()
        self.biadjacency = star_wars_villains(return_labels=False)
        self.house = house()

    def test_dir2undir(self):
        undirected_graph = directed2undirected(self.adjacency)
        self.assertEqual(undirected_graph.shape, self.adjacency.shape)
        self.assertTrue(is_symmetric(undirected_graph))
        error = 0.5 * directed2undirected(self.house) - self.house
        self.assertEqual(error.nnz, 0)

        n, m = self.adjacency.shape
        slr = SparseLR(self.adjacency, [(np.ones(n), np.ones(m))])
        undirected_graph = directed2undirected(slr)
        self.assertTrue(type(undirected_graph) == SparseLR)

    def test_bip2dir(self):
        n_nodes = self.biadjacency.shape[0] + self.biadjacency.shape[1]
        directed_graph = bipartite2directed(self.biadjacency)
        self.assertEqual(directed_graph.shape, (n_nodes, n_nodes))

        n, m = self.biadjacency.shape
        slr = SparseLR(self.biadjacency, [(np.ones(n), np.ones(m))])
        directed_graph = bipartite2directed(slr)
        self.assertTrue(type(directed_graph) == SparseLR)

    def test_bip2undir(self):
        n_nodes = self.biadjacency.shape[0] + self.biadjacency.shape[1]
        undirected_graph = bipartite2undirected(self.biadjacency)
        self.assertEqual(undirected_graph.shape, (n_nodes, n_nodes))
        self.assertTrue(is_symmetric(undirected_graph))

        n, m = self.biadjacency.shape
        slr = SparseLR(self.biadjacency, [(np.ones(n), np.ones(m))])
        undirected_graph = bipartite2undirected(slr)
        self.assertTrue(type(undirected_graph) == SparseLR)
