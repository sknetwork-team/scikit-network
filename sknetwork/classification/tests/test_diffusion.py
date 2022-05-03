#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for DiffusionClassifier"""

import unittest

from sknetwork.classification import DiffusionClassifier, DirichletClassifier
from sknetwork.data.test_graphs import *


class TestDiffusionClassifier(unittest.TestCase):

    def test_bipartite(self):
        biadjacency = test_bigraph()
        n_row, n_col = biadjacency.shape
        seeds_row = {0: 0, 1: 1}
        seeds_col = {5: 1}
        algos = [DiffusionClassifier(), DirichletClassifier()]
        for algo in algos:
            algo.fit(biadjacency, seeds_row=seeds_row, seeds_col=seeds_col)
            self.assertTrue(len(algo.labels_row_) == n_row)
            self.assertTrue(len(algo.labels_col_) == n_col)
        algos = [DiffusionClassifier(centering=False), DirichletClassifier(centering=False)]
        for algo in algos:
            algo.fit(biadjacency, seeds_row=seeds_row, seeds_col=seeds_col)
            self.assertTrue(len(algo.labels_row_) == n_row)
            self.assertTrue(len(algo.labels_col_) == n_col)

    def test_parallel(self):
        adjacency = test_graph()
        seeds = {0: 0, 1: 1}

        clf1 = DiffusionClassifier(n_jobs=None)
        clf2 = DiffusionClassifier(n_jobs=-1)

        labels1 = clf1.fit_transform(adjacency, seeds)
        labels2 = clf2.fit_transform(adjacency, seeds)

        self.assertTrue(np.allclose(labels1, labels2))
