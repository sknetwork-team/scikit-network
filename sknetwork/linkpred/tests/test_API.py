#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for link prediction API"""
import unittest
from abc import ABC
from typing import Iterable

import numpy as np

from sknetwork.data import house
from sknetwork.linkpred.base import BaseLinkPred


class Dummy(BaseLinkPred, ABC):
    """Dummy algorithm for testing purposes"""
    def __init__(self):
        super(Dummy, self).__init__()

    def fit(self, *args, **kwargs):
        """Dummy fit method"""
        return self

    def _predict_base(self, source: int, targets: Iterable):
        return np.array([1 for _ in targets])


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
        algo = Dummy().fit(adjacency)
        query = 'toto'
        with self.assertRaises(ValueError):
            algo.predict(query)
        query = np.ones((2, 3))
        with self.assertRaises(ValueError):
            algo.predict(query)
        query = [(0, 1), (2, 3)]
        pred = algo.predict(query)
        self.assertEqual(pred.shape, (len(query),))
