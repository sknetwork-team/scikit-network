#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for svd"""

import unittest

import numpy as np

from sknetwork.data import star_wars
from sknetwork.embedding import GSVD, SVD, PCA
from sknetwork.linalg import LanczosSVD


class TestSVD(unittest.TestCase):

    def test_options(self):
        biadjacency = star_wars(metadata=False)
        n_row, n_col = biadjacency.shape
        min_dim = min(n_row, n_col) - 1
        gsvd = GSVD(n_components=5, regularization=0., solver='halko')

        with self.assertWarns(Warning):
            gsvd.fit(biadjacency)
        self.assertEqual(gsvd.embedding_row_.shape, (n_row, min_dim))
        self.assertEqual(gsvd.embedding_col_.shape, (n_col, min_dim))

        embedding = gsvd.predict(np.array([0, 1, 1]))
        self.assertEqual(embedding.shape, (min_dim,))

        gsvd = GSVD(n_components=1, regularization=0.1, solver='lanczos')
        gsvd.fit(biadjacency)
        self.assertEqual(gsvd.embedding_row_.shape, (n_row, 1))

        pca = PCA(n_components=min_dim, solver='lanczos')
        pca.fit(biadjacency)
        self.assertEqual(pca.embedding_row_.shape, (n_row, min_dim))
        pca = PCA(n_components=min_dim, solver=LanczosSVD())
        pca.fit(biadjacency)
        self.assertEqual(pca.embedding_row_.shape, (n_row, min_dim))

        svd = SVD(n_components=min_dim, solver=LanczosSVD())
        svd.fit(biadjacency)
        self.assertEqual(svd.embedding_row_.shape, (n_row, min_dim))
