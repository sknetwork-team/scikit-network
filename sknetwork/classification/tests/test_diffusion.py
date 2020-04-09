#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for DiffusionClassifier"""

import unittest

from sknetwork.classification import DiffusionClassifier
from sknetwork.data.test_graphs import *


class TestDiffusionClassifier(unittest.TestCase):

    def test_parallel(self):
        adjacency = test_graph()
        seeds = {0: 0, 1: 1}

        clf1 = DiffusionClassifier(n_jobs=None)
        clf2 = DiffusionClassifier(n_jobs=-1)

        labels1 = clf1.fit_transform(adjacency, seeds)
        labels2 = clf2.fit_transform(adjacency, seeds)

        self.assertTrue(np.allclose(labels1, labels2))
