#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in March 2020
@author: Quentin Lutz <qlutz@enst.fr>
@author: Thomas Bonald <tbonald@enst.fr>
"""

import unittest

from sknetwork.data.test_graphs import *
from sknetwork.hierarchy import LouvainIteration, LouvainHierarchy, Paris


class TestLouvainHierarchy(unittest.TestCase):

    def test(self):
        louvain_iteration = LouvainIteration()
        louvain_iteration_ = LouvainIteration(resolution=2, depth=1)
        louvain_hierarchy = LouvainHierarchy()
        louvain_hierarchy_ = LouvainHierarchy(tol_aggregation=0.1)
        paris = Paris()
        paris_ = Paris(weights='uniform', reorder=False)
        for algo in [louvain_iteration, louvain_iteration_, louvain_hierarchy, louvain_hierarchy_, paris, paris_]:
            for input_matrix in [test_graph(), test_digraph(), test_bigraph()]:
                dendrogram = algo.fit_predict(input_matrix)
                self.assertEqual(dendrogram.shape, (input_matrix.shape[0] - 1, 4))
                if algo.bipartite:
                    self.assertEqual(algo.dendrogram_full_.shape, (sum(input_matrix.shape) - 1, 4))
        adjacency = test_graph()
        algo = Paris()
        dendrogram = algo.fit_predict(adjacency)
        dendrogram_ = algo.predict()
        self.assertAlmostEqual(np.linalg.norm(dendrogram - dendrogram_), 0)
