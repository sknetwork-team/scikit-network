#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for closeness.py"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.ranking.closeness import Closeness
from sknetwork.data.graph_data import house


# noinspection DuplicatedCode,PyMissingOrEmptyDocstring
class TestDiffusion(unittest.TestCase):

    def test_approximate(self):
        self.closeness = Closeness(method='approximate', n_jobs=-1)
        scores = self.closeness.fit_transform(house())
        self.assertEqual((np.round(scores, 2) == [0.67, 0.8, 0.67, 0.67, 0.8]).sum(), 5)

    def test_connected(self):
        self.closeness = Closeness()
        adjacency = sparse.identity(2, format='csr')
        with self.assertRaises(ValueError):
            self.closeness.fit(adjacency)
