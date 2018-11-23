# -*- coding: utf-8 -*-
# test for metrics.py
#
# Copyright 2018 Scikit-network Developers.
# Copyright 2018 Nathan de Lara <ndelara@enst.fr>
#
# This file is part of Scikit-network.

import unittest
import numpy as np
from scipy import sparse
from sknetwork.clustering.metrics import modularity, cocitation_modularity, performance, cocitation_performance


class TestClusteringMetrics(unittest.TestCase):

    def setUp(self):
        self.graph = sparse.csr_matrix(np.array([[0, 1, 1, 1],
                                                 [1, 0, 0, 0],
                                                 [1, 0, 0, 1],
                                                 [1, 0, 1, 0]]))
        self.labels = np.array([0, 1, 0, 0])
        self.unique_cluster = np.zeros(4, dtype=int)

    def test_modularity(self):
        self.assertAlmostEqual(modularity(self.graph, self.labels), -0.0312, 3)
        self.assertAlmostEqual(modularity(self.graph, self.unique_cluster), 0.)

    def test_cocitation_modularity(self):
        self.assertAlmostEqual(cocitation_modularity(self.graph, self.labels), 0.0521, 3)
        self.assertAlmostEqual(cocitation_modularity(self.graph, self.unique_cluster), 0.)

    def test_performance(self):
        self.assertAlmostEqual(performance(self.graph, self.labels), 0.625, 3)
        self.assertAlmostEqual(performance(self.graph, self.unique_cluster), 0.5)

    def test_cocitation_performance(self):
        self.assertAlmostEqual(cocitation_performance(self.graph, self.labels), 0.827, 3)
        self.assertAlmostEqual(cocitation_performance(self.graph, self.unique_cluster), 0.741, 3)
