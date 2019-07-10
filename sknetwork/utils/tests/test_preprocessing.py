# -*- coding: utf-8 -*-
# tests for metrics.py
""""tests for preprocessing.py"""

import unittest
import numpy as np
from sknetwork.utils.preprocessing import largest_connected_component


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.adjacency = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        self.biadjacency = np.array([[0, 1], [1, 0], [0, 1]])

    def test_largest_cc(self):
        largest_cc, indices = largest_connected_component(self.adjacency, return_labels=True)
        self.assertAlmostEqual(np.linalg.norm(largest_cc.toarray() - np.array([[0, 1], [1, 0]])), 0)
        self.assertEqual(np.linalg.norm(indices - np.array([0, 2])), 0)
        largest_cc, indices = largest_connected_component(self.biadjacency, return_labels=True)
        self.assertAlmostEqual(np.linalg.norm(largest_cc.toarray() - np.array([[1], [1]])), 0)
        self.assertEqual(np.linalg.norm(indices[0] - np.array([0, 2])), 0)
        self.assertEqual(np.linalg.norm(indices[1] - np.array([1])), 0)
