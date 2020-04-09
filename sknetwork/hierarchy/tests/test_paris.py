#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
@author: Quentin Lutz <qlutz@enst.fr>
"""

import unittest

from sknetwork.hierarchy import Paris, BiParis
from sknetwork.data.test_graphs import *


# noinspection PyMissingOrEmptyDocstring
class TestParis(unittest.TestCase):

    def setUp(self):
        self.paris = [Paris()]
        self.biparis = [BiParis()]

    def test_undirected(self):
        adjacency = test_graph()
        n = adjacency.shape[0]
        for paris in self.paris:
            dendrogram = paris.fit_transform(adjacency)
            self.assertEqual(dendrogram.shape, (n - 1, 4))

    def test_directed(self):
        adjacency = test_digraph()
        for paris in self.paris:
            dendrogram = paris.fit_transform(adjacency)
            n = adjacency.shape[0]
            self.assertEqual(dendrogram.shape, (n - 1, 4))

    def test_bipartite(self):
        biadjacency = test_bigraph()
        for biparis in self.biparis:
            biparis.fit(biadjacency)
            n1, n2 = biadjacency.shape
            self.assertEqual(biparis.dendrogram_.shape, (n1 - 1, 4))
            self.assertEqual(biparis.dendrogram_row_.shape, (n1 - 1, 4))
            self.assertEqual(biparis.dendrogram_col_.shape, (n2 - 1, 4))
            self.assertEqual(biparis.dendrogram_full_.shape, (n1 + n2 - 1, 4))

    def test_disconnected(self):
        adjacency = test_graph_disconnect()
        paris = Paris()
        dendrogram = paris.fit_transform(adjacency)
        self.assertEqual(dendrogram.shape, (9, 4))

    def test_options(self):
        paris = Paris(weights='uniform')
        adjacency = test_graph()
        dendrogram = paris.fit_transform(adjacency)
        n = adjacency.shape[0]
        self.assertEqual(dendrogram.shape, (n - 1, 4))


