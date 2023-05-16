#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for hierarchy API"""
import unittest

from sknetwork.data.test_graphs import *
from sknetwork.hierarchy import *


class TestHierarchyAPI(unittest.TestCase):

    def test_undirected(self):
        adjacency = test_graph()
        n = adjacency.shape[0]

        for algo in [Paris(), LouvainIteration()]:
            dendrogram = algo.fit_predict(adjacency)
            self.assertTupleEqual(dendrogram.shape, (n - 1, 4))

    def test_disconnected(self):
        adjacency = test_disconnected_graph()
        for algo in [Paris(), LouvainIteration()]:
            dendrogram = algo.fit_transform(adjacency)
            self.assertEqual(dendrogram.shape, (9, 4))
