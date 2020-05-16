#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for link prediction APi"""
import unittest

import numpy as np

from sknetwork.data import house
from sknetwork.linkpred import CommonNeighbors
from sknetwork.linkpred.base import BaseLinkPred


class TestLinkPred(unittest.TestCase):

    def test_base(self):
        algo = BaseLinkPred()
        query = 0
        with self.assertRaises(NotImplementedError):
            algo.predict(query)
        query = (0, 1)
        with self.assertRaises(NotImplementedError):
            algo.predict(query)

    def test_query(self):
        adjacency = house()
        algo = CommonNeighbors().fit(adjacency)
        query = 'toto'
        with self.assertRaises(ValueError):
            algo.predict(query)
        query = np.ones((2, 3))
        with self.assertRaises(ValueError):
            algo.predict(query)
