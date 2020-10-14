#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for KNN"""
import unittest

import numpy as np

from sknetwork.classification import KNN
from sknetwork.data import karate_club


class TestDiffusionClassifier(unittest.TestCase):

    def test_parallel(self):
        adjacency = karate_club(metadata=False)
        seeds = {0: 0, 1: 1}

        clf1 = KNN(n_neighbors=1, n_jobs=None)
        clf2 = KNN(n_neighbors=1, n_jobs=-1)

        labels1 = clf1.fit_transform(adjacency, seeds)
        labels2 = clf2.fit_transform(adjacency, seeds)

        self.assertTrue(np.allclose(labels1, labels2))
