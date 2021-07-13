#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2020
@author: Quentin Lutz <qlutz@enst.fr>
@author: Thomas Bonald <tbonald@enst.fr>
"""

import unittest

from sknetwork.data.test_graphs import test_graph, test_digraph, test_bigraph
from sknetwork.hierarchy.louvain_hierarchy import LouvainHierarchy


class TestLouvainHierarchy(unittest.TestCase):

    def test_options(self):
        for algo in [LouvainHierarchy(), LouvainHierarchy(resolution=2, depth=1)]:
            for input_matrix in [test_graph(), test_digraph(), test_bigraph()]:
                dendrogram = algo.fit_transform(input_matrix)
                n = input_matrix.shape[0]
                self.assertEqual(dendrogram.shape, (n - 1, 4))
                if algo.bipartite:
                    n = sum(input_matrix.shape)
                    self.assertEqual(algo.dendrogram_full_.shape, (n - 1, 4))
