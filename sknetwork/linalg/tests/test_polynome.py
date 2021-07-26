#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for polynomials."""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.data.test_graphs import test_graph
from sknetwork.linalg import Polynome


class TestPolynome(unittest.TestCase):

    def test_init(self):
        adjacency = test_graph()
        with self.assertRaises(ValueError):
            Polynome(adjacency, np.array([]))

    def test_operations(self):
        adjacency = test_graph()
        n = adjacency.shape[0]
        polynome = Polynome(adjacency, np.arange(3))
        x = np.random.randn(n)

        y1 = (polynome * 2).dot(x)
        y2 = (-polynome).dot(x)
        self.assertAlmostEqual(np.linalg.norm(0.5 * y1 + y2), 0)

    def test_dot(self):
        adjacency = sparse.eye(5, format='csr')
        polynome = Polynome(adjacency, np.arange(2))

        x = np.random.randn(5, 3)
        y = polynome.dot(x)
        self.assertAlmostEqual(np.linalg.norm(x - y), 0)

