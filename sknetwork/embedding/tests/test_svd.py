#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for svd"""

import unittest

import numpy as np

from sknetwork.data import star_wars
from sknetwork.embedding import GSVD


class TestSVD(unittest.TestCase):

    def test_options(self):
        biadjacency = star_wars(metadata=False)
        n_row, n_col = biadjacency.shape
        min_dim = min(n_row, n_col) - 1
        gsvd = GSVD(n_components=5, regularization=0., solver='halko', relative_regularization=True)

        with self.assertWarns(Warning):
            gsvd.fit(biadjacency)
        self.assertEqual(gsvd.embedding_row_.shape, (n_row, min_dim))
        self.assertEqual(gsvd.embedding_col_.shape, (n_col, min_dim))

        gsvd = GSVD(n_components=1, regularization=0.1, solver='halko', relative_regularization=True)
        gsvd.fit(biadjacency)
        gsvd.predict(np.random.rand(n_col))
