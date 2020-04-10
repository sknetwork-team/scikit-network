#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for hierarchy API"""

import unittest

from sknetwork.hierarchy import *
from sknetwork.data.test_graphs import *
from sknetwork.embedding import GSVD


class TestHierarchyAPI(unittest.TestCase):

    def test_undirected(self):
        adjacency = test_graph()
        n = adjacency.shape[0]

        hierarchy = [Paris(), Ward(GSVD(3)), LouvainHierarchy()]
        for hier_algo in hierarchy:
            dendrogram = hier_algo.fit_transform(adjacency)
            self.assertTupleEqual(dendrogram.shape, (n - 1, 4))

    def test_bipartite(self):
        biadjacency = test_bigraph()
        n_row, n_col = biadjacency.shape

        hierarchy = [BiParis(), BiWard(GSVD(3), cluster_col=True)]
        for hier_algo in hierarchy:
            hier_algo.fit_transform(biadjacency)
            dendrogram_row = hier_algo.dendrogram_row_
            self.assertTupleEqual(dendrogram_row.shape, (n_row - 1, 4))
            dendrogram_col = hier_algo.dendrogram_col_
            self.assertTupleEqual(dendrogram_col.shape, (n_col - 1, 4))
