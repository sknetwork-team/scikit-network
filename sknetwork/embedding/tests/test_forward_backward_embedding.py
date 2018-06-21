# -*- coding: utf-8 -*-
# test for metrics.py
#
# Copyright 2018 Scikit-network Developers.
# Copyright 2018 Nathan de Lara <ndelara@enst.fr>
#
# This file is part of Scikit-network.
#
# NetworkX is distributed under a BSD license; see LICENSE.txt for more information.

import unittest
import networkx as nx
from sknetwork.embedding.forward_backward_embedding import *


class TestForwardBackwardEmbedding(unittest.TestCase):

    def setUp(self):
        self.embedding = ForwardBackwardEmbedding(10)

    def test_directed_embedding(self):
        graph = nx.fast_gnp_random_graph(100, 0.05, directed=True)
        self.embedding.fit(graph)
        self.assertEqual(self.embedding.leftsvd_.shape, (100, 10))

    def test_undirected_embedding(self):
        graph = nx.fast_gnp_random_graph(100, 0.05, directed=False)
        self.embedding.fit(graph)
        self.assertEqual(self.embedding.leftsvd_.shape, (100, 10))

    def test_bipartite_embedding(self):
        graph = nx.algorithms.bipartite.random_graph(50, 200, 0.01)
        self.embedding.fit(graph, (range(50), range(50, 250)))
        self.assertEqual(self.embedding.leftsvd_.shape, (50, 10))
        self.assertEqual(self.embedding.rightsvd_.shape, (200, 10))
