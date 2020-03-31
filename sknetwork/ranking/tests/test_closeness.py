#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for closeness.py"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.ranking.closeness import Closeness
from sknetwork.data.test_graphs import *


# noinspection DuplicatedCode,PyMissingOrEmptyDocstring
class TestDiffusion(unittest.TestCase):

    def test_undirected(self):
        adjacency = Simple().adjacency
        closeness = Closeness(method='approximate', n_jobs=-1)
        scores = closeness.fit_transform(adjacency)
        self.assertEqual(len(scores), 10)

    def test_disconnected(self):
        adjacency = DisSimple().adjacency
        closeness = Closeness()
        with self.assertRaises(ValueError):
            closeness.fit(adjacency)
