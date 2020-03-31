#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for shortest_path.py"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.basics import shortest_path
from sknetwork.data import karate_club


class TestShortestPath(unittest.TestCase):

    def test_parallel(self):
        adjacency: sparse.csr_matrix = karate_club()
        truth = shortest_path(adjacency)
        parallel = shortest_path(adjacency, n_jobs=None)
        self.assertEqual(np.equal(truth, parallel).sum(), 34 ** 2)
