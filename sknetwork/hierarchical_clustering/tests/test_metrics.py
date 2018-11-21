# -*- coding: utf-8 -*-
# test for metrics.py
#
# Copyright 2018 Scikit-network Developers.
# Copyright 2018 Bertrand Charpentier <bertrand.charpentier@live.fr>
#
# This file is part of Scikit-network.
#
# NetworkX is distributed under a BSD license; see LICENSE.txt for more information.

import unittest
import networkx as nx
from sknetwork.hierarchical_clustering.agglomerative_clustering import linkage_clustering
from sknetwork.hierarchical_clustering.metrics import hierarchical_cost


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.weighted_graph = nx.Graph()
        self.weighted_graph.add_nodes_from([0, 1, 2, 3, 4, 5])
        self.weighted_graph.add_weighted_edges_from([(0, 1, 2), (0, 2, 1.5), (1, 2, 1.5), (2, 3, 1), (3, 4, 1.5),
                                                     (3, 5, 1.5), (4, 5, 2)])
        self.d_w = linkage_clustering(self.weighted_graph)

        self.unitary_weighted_graph = nx.Graph()
        self.unitary_weighted_graph.add_nodes_from([0, 1, 2, 3, 4, 5])
        self.unitary_weighted_graph.add_weighted_edges_from([(0, 1, 1), (0, 2, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (3, 5, 1),
                                                             (4, 5, 1)])
        self.d_wu = linkage_clustering(self.weighted_graph)

        self.unweighted_graph = nx.Graph()
        self.unweighted_graph.add_nodes_from([0, 1, 2, 3, 4, 5])
        self.unweighted_graph.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)])
        self.d_u = linkage_clustering(self.weighted_graph, affinity='unitary')

    def test_unknown_options(self):
        with self.assertRaises(ValueError):
            hierarchical_cost(self.weighted_graph, self.d_w, affinity='unknown')

        with self.assertRaises(ValueError):
            hierarchical_cost(self.weighted_graph, self.d_w, linkage='unknown')

    def test_classic_linkage(self):
        s_w_u = hierarchical_cost(self.weighted_graph, self.d_w, affinity='unitary', linkage='classic')
        s_w_w = hierarchical_cost(self.weighted_graph, self.d_w, affinity='weighted', linkage='classic')

        self.assertNotEqual(s_w_u, s_w_w)

        s_wu_u = hierarchical_cost(self.unitary_weighted_graph, self.d_wu, affinity='unitary', linkage='classic')
        s_wu_w = hierarchical_cost(self.unitary_weighted_graph, self.d_wu, affinity='weighted', linkage='classic')

        self.assertEqual(s_wu_u, s_wu_w)

        with self.assertRaises(KeyError):
            hierarchical_cost(self.unweighted_graph, self.d_u, affinity='weighted', linkage='classic')
