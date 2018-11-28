# -*- coding: utf-8 -*-
# test for agglomerative_clustering.py
#
# Copyright 2018 Scikit-network Developers.
# Copyright 2018 Bertrand Charpentier <bertrand.charpentier@live.fr>
#
# This file is part of Scikit-network.
#
# NetworkX is distributed under a BSD license; see LICENSE.txt for more information.

import unittest
import networkx as nx
import numpy as np
from sknetwork.hierarchy.agglomerative_clustering import linkage_clustering


class TestAgglomerativeClustering(unittest.TestCase):

    def setUp(self):
        self.weighted_graph = nx.Graph()
        self.weighted_graph.add_nodes_from([0, 1, 2, 3, 4, 5])
        self.weighted_graph.add_weighted_edges_from([(0, 1, 2), (0, 2, 1.5), (1, 2, 1.5), (2, 3, 1), (3, 4, 1.5),
                                                     (3, 5, 1.5), (4, 5, 2)])

        self.unitary_weighted_graph = nx.Graph()
        self.unitary_weighted_graph.add_nodes_from([0, 1, 2, 3, 4, 5])
        self.unitary_weighted_graph.add_weighted_edges_from([(0, 1, 1), (0, 2, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1),
                                                             (3, 5, 1), (4, 5, 1)])
        self.unweighted_graph = nx.Graph()
        self.unweighted_graph.add_nodes_from([0, 1, 2, 3, 4, 5])
        self.unweighted_graph.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)])

    def test_unknown_options(self):
        with self.assertRaises(ValueError):
            linkage_clustering(self.weighted_graph, affinity='unknown')

        with self.assertRaises(ValueError):
            linkage_clustering(self.weighted_graph, linkage='unknown')

    def test_single_linkage(self):
        d_w_u = linkage_clustering(self.weighted_graph, affinity='unitary', linkage='single',
                                   f=lambda l: - np.log(l), check=True)
        d_w_w = linkage_clustering(self.weighted_graph, affinity='weighted', linkage='single',
                                   f=lambda l: - np.log(l), check=True)

        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(d_w_u, d_w_w)

        d_wu_u = linkage_clustering(self.unitary_weighted_graph, affinity='unitary', linkage='single',
                                    f=lambda l: - np.log(l), check=True)
        d_wu_w = linkage_clustering(self.unitary_weighted_graph, affinity='weighted', linkage='single',
                                    f=lambda l: - np.log(l), check=True)

        np.testing.assert_array_equal(d_w_u, d_wu_u)
        np.testing.assert_array_equal(d_w_u, d_wu_w)

        d_u_u = linkage_clustering(self.unweighted_graph, affinity='unitary', linkage='single',
                                   f=lambda l: - np.log(l), check=True)
        with self.assertRaises(KeyError):
            linkage_clustering(self.unweighted_graph, affinity='weighted', linkage='single',
                               f=lambda l: - np.log(l), check=True)

        np.testing.assert_array_equal(d_w_u, d_u_u)

    def test_average_linkage(self):
        d_w_u = linkage_clustering(self.weighted_graph, affinity='unitary', linkage='average',
                                   f=lambda l: - np.log(l), check=True)
        d_w_w = linkage_clustering(self.weighted_graph, affinity='weighted', linkage='average',
                                   f=lambda l: - np.log(l), check=True)

        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(d_w_u, d_w_w)

        d_wu_u = linkage_clustering(self.unitary_weighted_graph, affinity='unitary', linkage='average',
                                    f=lambda l: - np.log(l), check=True)
        d_wu_w = linkage_clustering(self.unitary_weighted_graph, affinity='weighted', linkage='average',
                                    f=lambda l: - np.log(l), check=True)

        np.testing.assert_array_equal(d_w_u, d_wu_u)
        np.testing.assert_array_equal(d_w_u, d_wu_w)

        d_u_u = linkage_clustering(self.unweighted_graph, affinity='unitary', linkage='average',
                                   f=lambda l: - np.log(l), check=True)
        with self.assertRaises(KeyError):
            linkage_clustering(self.unweighted_graph, affinity='weighted', linkage='average',
                               f=lambda l: - np.log(l), check=True)

        np.testing.assert_array_equal(d_w_u, d_u_u)

    def test_complete_linkage(self):
        d_w_u = linkage_clustering(self.weighted_graph, affinity='unitary', linkage='complete',
                                   f=lambda l: - np.log(l), check=True)
        d_w_w = linkage_clustering(self.weighted_graph, affinity='weighted', linkage='complete',
                                   f=lambda l: - np.log(l), check=True)

        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(d_w_u, d_w_w)

        d_wu_u = linkage_clustering(self.unitary_weighted_graph, affinity='unitary', linkage='complete',
                                    f=lambda l: - np.log(l), check=True)
        d_wu_w = linkage_clustering(self.unitary_weighted_graph, affinity='weighted', linkage='complete',
                                    f=lambda l: - np.log(l), check=True)

        np.testing.assert_array_equal(d_w_u, d_wu_u)
        np.testing.assert_array_equal(d_w_u, d_wu_w)

        d_u_u = linkage_clustering(self.unweighted_graph, affinity='unitary', linkage='complete',
                                   f=lambda l: - np.log(l), check=True)
        with self.assertRaises(KeyError):
            linkage_clustering(self.unweighted_graph, affinity='weighted', linkage='complete',
                               f=lambda l: - np.log(l), check=True)

        np.testing.assert_array_equal(d_w_u, d_u_u)

    def test_modular_linkage(self):
        d_w_u = linkage_clustering(self.weighted_graph, affinity='unitary', linkage='modular',
                                   f=lambda l: - np.log(l), check=True)
        d_w_w = linkage_clustering(self.weighted_graph, affinity='weighted', linkage='modular',
                                   f=lambda l: - np.log(l), check=True)

        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(d_w_u, d_w_w)

        d_wu_u = linkage_clustering(self.unitary_weighted_graph, affinity='unitary', linkage='modular',
                                    f=lambda l: - np.log(l), check=True)
        d_wu_w = linkage_clustering(self.unitary_weighted_graph, affinity='weighted', linkage='modular',
                                    f=lambda l: - np.log(l), check=True)

        np.testing.assert_array_equal(d_w_u, d_wu_u)
        np.testing.assert_array_equal(d_w_u, d_wu_w)

        d_u_u = linkage_clustering(self.unweighted_graph, affinity='unitary', linkage='modular',
                                   f=lambda l: - np.log(l), check=True)
        with self.assertRaises(KeyError):
            linkage_clustering(self.unweighted_graph, affinity='weighted', linkage='modular',
                               f=lambda l: - np.log(l), check=True)

        np.testing.assert_array_equal(d_w_u, d_u_u)
