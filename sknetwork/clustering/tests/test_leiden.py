#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Leiden"""
import unittest

from sknetwork.clustering import Leiden
from sknetwork.data.test_graphs import *
from sknetwork.utils import bipartite2undirected


class TestLeidenClustering(unittest.TestCase):

    def test_disconnected(self):
        adjacency = test_disconnected_graph()
        n = adjacency.shape[0]
        labels = Leiden().fit_predict(adjacency)
        self.assertEqual(len(labels), n)

    def test_modularity(self):
        adjacency = test_graph()
        leiden_d = Leiden(modularity='dugue')
        leiden_n = Leiden(modularity='newman')
        labels_d = leiden_d.fit_predict(adjacency)
        labels_n = leiden_n.fit_predict(adjacency)
        self.assertTrue((labels_d == labels_n).all())

    def test_bipartite(self):
        biadjacency = test_bigraph()
        adjacency = bipartite2undirected(biadjacency)
        leiden = Leiden(modularity='newman')
        labels1 = leiden.fit_predict(adjacency)
        leiden.fit(biadjacency)
        labels2 = np.concatenate((leiden.labels_row_, leiden.labels_col_))
        self.assertTrue((labels1 == labels2).all())
