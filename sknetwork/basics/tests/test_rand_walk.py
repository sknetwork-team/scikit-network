#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for rand_walk.py"""

import unittest

import numpy as np

from sknetwork.basics import transition_matrix


class TestRandomWalk(unittest.TestCase):

    def test_transition(self):
        self.adjacency = np.array([[0, 1], [0, 0]])

        trans_matrix = transition_matrix(self.adjacency)
        delta = self.adjacency - trans_matrix.todense()
        self.assertEqual(0, np.linalg.norm(delta))

        trans_matrix = transition_matrix(self.adjacency.T)
        delta = self.adjacency.T - trans_matrix.todense()
        self.assertEqual(0, np.linalg.norm(delta))
