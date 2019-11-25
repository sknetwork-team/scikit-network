#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for shortest_path.py"""

import unittest

import numpy as np

from sknetwork.basics import shortest_path
from sknetwork.data import karate_club


class TestShortestPath(unittest.TestCase):

    def test_parallel(self):
        karate = karate_club()
        truth = shortest_path(karate)
        parallel = shortest_path(karate, n_jobs=None)
        self.assertEqual(np.equal(truth, parallel).sum(), 34 ** 2)
