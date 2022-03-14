#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for parse.py"""
import unittest
import numpy as np

from sknetwork.utils.parse import edgelist2adjacency, edgelist2biadjacency


class TestFormats(unittest.TestCase):

    def test_adjacency(self):
        edge_list = [(1, 2), (3, 4), (1, 0), (1, 2)]
        adjacency = edgelist2adjacency(edge_list)
        self.assertEqual(adjacency.nnz, 3)
        adjacency = edgelist2adjacency(edge_list, undirected=True)
        self.assertEqual(adjacency.nnz, 6)
        adjacency = edgelist2adjacency(edge_list, weighted=False)
        self.assertEqual(np.max(adjacency.data), 1)
        edge_list = [(1, 2, 1), (3, 4, 3), (1, 0, 1), (1, 2, 2)]
        adjacency = edgelist2adjacency(edge_list)
        self.assertEqual(adjacency.nnz, 3)
        self.assertEqual(adjacency[1, 2], 3)
        adjacency = edgelist2adjacency(edge_list, undirected=True)
        self.assertEqual(adjacency.nnz, 6)
        adjacency = edgelist2adjacency(edge_list, weighted=False)
        self.assertEqual(np.max(adjacency.data), 1)

    def test_biadjacency(self):
        edge_list = [(1, 2), (3, 4), (1, 0), (1, 2)]
        biadjacency = edgelist2biadjacency(edge_list)
        self.assertEqual(biadjacency.nnz, 3)
        biadjacency = edgelist2biadjacency(edge_list, weighted=False)
        self.assertEqual(np.max(biadjacency.data), 1)
        edge_list = [(1, 2, 1), (3, 4, 3), (1, 0, 1), (1, 2, 2)]
        biadjacency = edgelist2biadjacency(edge_list)
        self.assertEqual(biadjacency[1, 2], 3)
