#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.hierarchy import Paris, tree_sampling_divergence, dasgupta_score
from sknetwork.data import line_graph, karate_club


# noinspection PyMissingOrEmptyDocstring
class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.paris = Paris(engine='python')
        self.line_graph = line_graph()
        self.karate_club = karate_club()

    def test_undirected(self):
        adjacency: sparse.csr_matrix = self.karate_club
        self.paris.fit(adjacency)
        dendrogram = self.paris.dendrogram_
        dc = dasgupta_score(adjacency, dendrogram)
        self.assertAlmostEqual(dc, .666, 2)
        tsd = tree_sampling_divergence(adjacency, dendrogram, normalized=False)
        self.assertAlmostEqual(tsd, .717, 2)
        tsd = tree_sampling_divergence(adjacency, dendrogram)
        self.assertAlmostEqual(tsd, .487, 2)

    def test_disconnected(self):
        adjacency = np.eye(2)
        self.paris.fit(adjacency)
        dendrogram = self.paris.dendrogram_
        dc = dasgupta_score(adjacency, dendrogram)
        self.assertAlmostEqual(dc, 1, 2)
        tsd = tree_sampling_divergence(adjacency, dendrogram, normalized=False)
        self.assertAlmostEqual(tsd, 0, 2)
