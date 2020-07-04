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

    def test_disconnected(self):
        adjacency = test_graph_disconnect()
        hierarchy = [Paris(), Ward(GSVD(3)), LouvainHierarchy()]
        for hier_algo in hierarchy:
            dendrogram = hier_algo.fit_transform(adjacency)
            self.assertEqual(dendrogram.shape, (9, 4))

    def test_bipartite(self):
        biadjacency = test_bigraph()
        n_row, n_col = biadjacency.shape

        hierarchy = [BiParis(), BiWard(GSVD(3), cluster_col=True, cluster_both=True), BiLouvainHierarchy()]
        for algo in hierarchy:
            algo.fit_transform(biadjacency)
            self.assertTupleEqual(algo.dendrogram_row_.shape, (n_row - 1, 4))
            self.assertTupleEqual(algo.dendrogram_col_.shape, (n_col - 1, 4))
            if algo.dendrogram_full_ is not None:
                self.assertTupleEqual(algo.dendrogram_full_.shape, (n_row + n_col - 1, 4))
