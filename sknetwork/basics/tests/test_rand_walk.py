# -*- coding: utf-8 -*-
# tests for metrics.py
""""tests for rand_walk.py"""

import unittest

import numpy as np

from sknetwork.basics import transition_matrix


class TestRandomWalk(unittest.TestCase):

    def setUp(self):
        self.adjacency = np.array([[0, 1], [0, 0]])

    def test_transition(self):
        trans_matrix = transition_matrix(self.adjacency)
        delta = self.adjacency - trans_matrix.todense()
        self.assertEqual(0, np.linalg.norm(delta))

        trans_matrix = transition_matrix(self.adjacency.T)
        delta = self.adjacency.T - trans_matrix.todense()
        self.assertEqual(0, np.linalg.norm(delta))
