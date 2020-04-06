#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for PageRankClassifier"""

import unittest

from sknetwork.classification import PageRankClassifier
from sknetwork.data.test_graphs import *


class TestPageRankClassifier(unittest.TestCase):

    def test_solvers(self):
        adjacency = test_graph()
        seeds = {0: 0, 1: 1}

        clf1 = PageRankClassifier(solver='lsqr')
        clf2 = PageRankClassifier(solver='lanczos')
        clf3 = PageRankClassifier()

        labels1 = clf1.fit_transform(adjacency, seeds)
        labels2 = clf2.fit_transform(adjacency, seeds)
        labels3 = clf3.fit_transform(adjacency, seeds)

        self.assertTrue(np.allclose(labels1, labels2))
        self.assertTrue(np.allclose(labels2, labels3))
