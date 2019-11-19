#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for shortest_path.py"""

import unittest

import numpy as np

from sknetwork.basics import shortest_path
from sknetwork.toy_graphs import karate_club


class TestShortestPath(unittest.TestCase):

    def setUp(self):
        self.karate = karate_club()

    def test_parallel(self):
        truth = shortest_path(self.karate)
        parallel = shortest_path(self.karate, n_jobs=-1)
        self.assertEqual(np.equal(truth, parallel).sum(), 34**2)


