#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.hierarchy import Paris, tree_sampling_divergence, dasgupta_cost
from sknetwork.toy_graphs import random_graph, random_bipartite_graph, line_graph, karate_club


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.paris = Paris(engine='python')
        self.line_graph = line_graph()
        self.karate_club = karate_club()

    def test_undirected(self):
        adjacency: sparse.csr_matrix = self.karate_club
        self.paris.fit(adjacency)
        dendrogram = self.paris.dendrogram_
        dc = dasgupta_cost(adjacency, dendrogram, force_undirected=True)
        self.assertAlmostEqual(dc, .666, 2)
        tsd = tree_sampling_divergence(adjacency, dendrogram, force_undirected=True, normalized=False)
        self.assertAlmostEqual(tsd, .717, 2)
        tsd = tree_sampling_divergence(adjacency, dendrogram, force_undirected=True)
        self.assertAlmostEqual(tsd, .487, 2)

    def test_disconnected(self):
        adjacency = np.eye(2)
        self.paris.fit(adjacency)
        dendrogram = self.paris.dendrogram_
        dc = dasgupta_cost(adjacency, dendrogram)
        self.assertAlmostEqual(dc, 1, 2)
        tsd = tree_sampling_divergence(adjacency, dendrogram, normalized=False)
        self.assertAlmostEqual(tsd, 0, 2)


"""
    def test_directed(self):
        adjacency = self.line_graph
        self.paris.fit(adjacency)
        dendrogram = self.paris.dendrogram_
        dc = dasgupta_cost(adjacency, dendrogram)
        self.assertAlmostEqual(dc, .833, 2)
        tsd = tree_sampling_divergence(adjacency, dendrogram, normalized=False)
        self.assertAlmostEqual(tsd, .346, 2)
        tsd = tree_sampling_divergence(adjacency, dendrogram)
        self.assertAlmostEqual(tsd, .499, 2)

    def test_random(self):
        adjacency = self.random_graph
        self.paris.fit(adjacency)
        dendrogram = self.paris.dendrogram_
        dc = dasgupta_cost(adjacency, dendrogram)
        self.assertAlmostEqual(dc, 0.25, 2)
        tsd = tree_sampling_divergence(adjacency, dendrogram)
        self.assertAlmostEqual(tsd, 0.543, 2)

    def test_bipartite(self):
        biadjacency = self.random_bipartite_graph
        self.paris.fit(biadjacency)
        dendrogram = self.paris.dendrogram_
        dc = dasgupta_cost(biadjacency, dendrogram)
        self.assertAlmostEqual(dc, 0.342, 2)
        tsd = tree_sampling_divergence(biadjacency, dendrogram)
        self.assertAlmostEqual(tsd, 0.882, 2)
"""
