#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for shortest_path.py"""
import unittest

from sknetwork.data.test_graphs import *
from sknetwork.path.shortest_path import get_shortest_path


class TestShortestPath(unittest.TestCase):

    def test_path(self):
        adjacency = test_graph_empty()
        path = get_shortest_path(adjacency, 0)
        self.assertEqual(path.nnz, 0)

        adjacency = test_graph()
        path = get_shortest_path(adjacency, 0)
        self.assertEqual(path.nnz, 10)
        path = get_shortest_path(adjacency, [0, 4, 6])
        self.assertEqual(path.nnz, 10)
        path = get_shortest_path(adjacency, np.arange(10))
        self.assertEqual(path.nnz, 0)
        path = get_shortest_path(adjacency, [0, 5])
        self.assertEqual(path.nnz, 9)

        adjacency = test_disconnected_graph()
        path = get_shortest_path(adjacency, 4)
        self.assertEqual(path.nnz, 5)

        adjacency = test_digraph()
        path = get_shortest_path(adjacency, 0)
        self.assertEqual(path.nnz, 5)

        biadjacency = test_bigraph()
        path = get_shortest_path(biadjacency, 0)
        self.assertEqual(path.nnz, 3)
        self.assertTrue(path.shape[0] == np.sum(biadjacency.shape))
        path = get_shortest_path(biadjacency, source_col=np.arange(biadjacency.shape[1]))
        self.assertEqual(path.nnz, biadjacency.nnz)
