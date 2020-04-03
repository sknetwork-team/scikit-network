#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

from sknetwork.hierarchy import Ward, BiWard
from sknetwork.embedding import GSVD, Spectral
from sknetwork.data.test_graphs import *


class TestWard(unittest.TestCase):

    def setUp(self):
        self.ward = Ward(embedding_method=GSVD(3))
        self.biward = BiWard(embedding_method=GSVD(2), cluster_col=True, cluster_both=True)

    def test_undirected(self):
        adjacency = test_graph()
        dendrogram = self.ward.fit_transform(adjacency)
        self.assertEqual(dendrogram.shape, (adjacency.shape[0] - 1, 4))

    def test_directed(self):
        adjacency = test_digraph()
        dendrogram = self.ward.fit_transform(adjacency)
        self.assertEqual(dendrogram.shape, (adjacency.shape[0] - 1, 4))

    def test_bipartite(self):
        biadjacency = test_bigraph()
        self.biward.fit(biadjacency)
        n1, n2 = biadjacency.shape
        self.assertEqual(self.biward.dendrogram_.shape, (n1 - 1, 4))
        self.assertEqual(self.biward.dendrogram_row_.shape, (n1 - 1, 4))
        self.assertEqual(self.biward.dendrogram_col_.shape, (n2 - 1, 4))
        self.assertEqual(self.biward.dendrogram_full_.shape, (n1 + n2 - 1, 4))

    def test_disconnected(self):
        adjacency = test_graph_disconnect()
        dendrogram = self.ward.fit_transform(adjacency)
        self.assertEqual(dendrogram.shape, (adjacency.shape[0] - 1, 4))

    def test_options(self):
        adjacency = test_graph()
        ward = Ward(embedding_method=Spectral(3))
        dendrogram = ward.fit_transform(adjacency)
        self.assertEqual(dendrogram.shape, (adjacency.shape[0] - 1, 4))
