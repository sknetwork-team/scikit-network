#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for postprocessing"""

import unittest

import numpy as np

from sknetwork.ranking.postprocess import top_k


class TestPostprocessing(unittest.TestCase):

    def test_top_k(self):
        scores = np.arange(10)
        index = top_k(scores, 3)
        self.assertTrue(set(index) == {7, 8, 9})
        index = top_k(scores, 10)
        self.assertTrue(len(index) == 10)
        index = top_k(scores, 20)
        self.assertTrue(len(index) == 10)
