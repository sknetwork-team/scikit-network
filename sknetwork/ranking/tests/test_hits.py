#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for diffusion.py"""

import unittest

from sknetwork.ranking import HITS
from sknetwork.data.test_graphs import test_bigraph


class TestHITS(unittest.TestCase):

    def test_keywords(self):
        biadjacency = test_bigraph()
        n_row, n_col = biadjacency.shape

        for hits in [HITS(solver='lanczos', tol=0.1), HITS(solver='halko', n_oversamples=1)]:
            hits.fit(biadjacency)
            self.assertEqual(hits.scores_row_.shape, (n_row,))
            self.assertEqual(hits.scores_col_.shape, (n_col,))
