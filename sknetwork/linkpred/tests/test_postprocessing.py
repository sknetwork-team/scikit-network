#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for link prediction postprocessing"""
import unittest

import numpy as np

from sknetwork.data import house
from sknetwork.linkpred import is_edge


class TestLinkPredPostProcessing(unittest.TestCase):

    def test_signature(self):
        adjacency = house()

        with self.assertRaises(ValueError):
            is_edge(adjacency, 'toto')

        with self.assertRaises(ValueError):
            is_edge(adjacency, np.ones(8).reshape((2, 2, 2)))
