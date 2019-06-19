#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for adjacency_formats.py"""


import unittest
from sknetwork.toy_graphs import star_wars_villains_graph, rock_paper_scissors_graph, house_graph
from sknetwork.utils.adjacency_formats import *
from sknetwork.utils.checks import is_symmetric


class TestFormats(unittest.TestCase):

    def setUp(self):
        self.adjacency = rock_paper_scissors_graph()
        self.biadjacency = star_wars_villains_graph(return_labels=False)
        self.house = house_graph()

    def test_dir2undir(self):
        undirected_graph = directed2undirected(self.adjacency)
        self.assertEqual(undirected_graph.shape, self.adjacency.shape)
        self.assertTrue(is_symmetric(undirected_graph))
        error = 0.5 * directed2undirected(self.house) - self.house
        self.assertEqual(error.nnz, 0)

    def test_bip2dir(self):
        n_nodes = self.biadjacency.shape[0] + self.biadjacency.shape[1]
        directed_graph = bipartite2directed(self.biadjacency)
        self.assertEqual(directed_graph.shape, (n_nodes, n_nodes))

    def test_bip2undir(self):
        n_nodes = self.biadjacency.shape[0] + self.biadjacency.shape[1]
        undirected_graph = bipartite2undirected(self.biadjacency)
        self.assertEqual(undirected_graph.shape, (n_nodes, n_nodes))
        self.assertTrue(is_symmetric(undirected_graph))
