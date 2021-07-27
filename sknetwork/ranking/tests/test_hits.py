#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for his.py"""

import unittest

from sknetwork.data.test_graphs import test_bigraph
from sknetwork.ranking import HITS


class TestHITS(unittest.TestCase):

    def test_keywords(self):
        biadjacency = test_bigraph()
        n_row, n_col = biadjacency.shape

        hits = HITS()
        hits.fit(biadjacency)
        self.assertEqual(hits.scores_row_.shape, (n_row,))
        self.assertEqual(hits.scores_col_.shape, (n_col,))
