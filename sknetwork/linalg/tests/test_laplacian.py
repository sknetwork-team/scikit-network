#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for laplacian."""

import unittest

import numpy as np

from sknetwork.data.test_graphs import test_graph
from sknetwork.linalg import get_laplacian


class TestLaplacian(unittest.TestCase):

    def test(self):
        adjacency = test_graph()
        laplacian = get_laplacian(adjacency)
        self.assertEqual(np.linalg.norm(laplacian.dot(np.ones(adjacency.shape[0]))), 0)
