#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for distances.py"""
import unittest

from sknetwork.data.test_graphs import *
from sknetwork.path.distances import get_distances


class TestDistances(unittest.TestCase):

    def test_input(self):
        adjacency = test_graph()
        with self.assertRaises(ValueError):
            get_distances(adjacency)
        with self.assertRaises(ValueError):
            get_distances(adjacency, source=0, source_row=5)

    def test_algo(self):
        adjacency = test_graph()
        distances = get_distances(adjacency, 0)
        distances_ = np.array([0, 1, 3, 2, 2, 3, 2, 3, 4, 4])
        self.assertTrue(all(distances == distances_))
        distances = get_distances(adjacency, 0, transpose=True)
        self.assertTrue(all(distances == distances_))
        distances = get_distances(adjacency, [0, 5])
        distances_ = np.array([0, 1, 3, 2, 1, 0, 1, 2, 4, 3])
        self.assertTrue(all(distances == distances_))

        adjacency = test_graph_empty()
        source = [0, 3]
        distances = get_distances(adjacency, source)
        distances_ = -np.ones(len(distances), dtype=int)
        distances_[source] = 0
        self.assertTrue(all(distances == distances_))

        adjacency = test_digraph()
        distances = get_distances(adjacency, [0])
        distances_ = np.array([0, 1, 3, 2, 2, 3, -1, -1, -1, -1])
        self.assertTrue(all(distances == distances_))
        distances = get_distances(adjacency, [0], transpose=True)
        self.assertTrue(sum(distances < 0) == 9)
        distances = get_distances(adjacency, [0, 5], transpose=True)
        distances_ = np.array([0, 2, -1, -1, 1, 0, 1, -1, -1, -1])
        self.assertTrue(all(distances == distances_))

        biadjacency = test_bigraph()
        distances_row, distances_col = get_distances(biadjacency, [0])
        distances_row_, distances_col_ = np.array([0, -1, 2, -1, -1, -1]), np.array([3,  1, -1, -1, -1, -1, -1, -1])
        self.assertTrue(all(distances_row == distances_row_))
        self.assertTrue(all(distances_col == distances_col_))

        adjacency = test_graph()
        distances_row, distances_col = get_distances(adjacency, source_col=[0])
        self.assertTrue(all(distances_row % 2))
        self.assertTrue(all((distances_col + 1) % 2))

        biadjacency = test_bigraph()
        distances_row, distances_col = get_distances(biadjacency, source=0, source_col=[1, 2])
        distances_row_, distances_col_ = np.array([0,  1,  1, -1, -1, -1]), np.array([2,  0,  0,  2, -1, -1, -1, -1])
        self.assertTrue(all(distances_row == distances_row_))
        self.assertTrue(all(distances_col == distances_col_))
