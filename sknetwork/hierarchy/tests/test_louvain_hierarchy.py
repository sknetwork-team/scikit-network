#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2020
@author: Quentin Lutz <qlutz@enst.fr>
@author: Thomas Bonald <tbonald@enst.fr>
"""

import unittest

from sknetwork.hierarchy.louvain_hierarchy import LouvainHierarchy, BiLouvainHierarchy
from sknetwork.data.test_graphs import *


# noinspection PyMissingOrEmptyDocstring
class TestLouvainHierarchy(unittest.TestCase):

    def setUp(self):
        self.louvain = [LouvainHierarchy()]
        self.bilouvain = [BiLouvainHierarchy()]

    def test_undirected(self):
        adjacency = test_graph()
        n = adjacency.shape[0]
        for louvain in self.louvain:
            dendrogram = louvain.fit_transform(adjacency)
            self.assertEqual(dendrogram.shape, (n - 1, 4))

    def test_directed(self):
        adjacency = test_digraph()
        for louvain in self.louvain:
            dendrogram = louvain.fit_transform(adjacency)
            n = adjacency.shape[0]
            self.assertEqual(dendrogram.shape, (n - 1, 4))

    def test_bipartite(self):
        biadjacency = test_bigraph()
        for bilouvain in self.bilouvain:
            bilouvain.fit(biadjacency)
            n1, n2 = biadjacency.shape
            self.assertEqual(bilouvain.dendrogram_.shape, (n1 - 1, 4))
            self.assertEqual(bilouvain.dendrogram_row_.shape, (n1 - 1, 4))
            self.assertEqual(bilouvain.dendrogram_col_.shape, (n2 - 1, 4))
            self.assertEqual(bilouvain.dendrogram_full_.shape, (n1 + n2 - 1, 4))

    def test_disconnected(self):
        adjacency = test_graph_disconnect()
        louvain = LouvainHierarchy()
        dendrogram = louvain.fit_transform(adjacency)
        self.assertEqual(dendrogram.shape, (9, 4))

    def test_options(self):
        louvain = LouvainHierarchy(resolution=2)
        adjacency = test_graph()
        dendrogram = louvain.fit_transform(adjacency)
        n = adjacency.shape[0]
        self.assertEqual(dendrogram.shape, (n - 1, 4))
