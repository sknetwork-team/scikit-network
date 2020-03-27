#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for clustering post-processing"""

import unittest

import numpy as np

from sknetwork.clustering import reindex_clusters


class TestClusteringPostProcessing(unittest.TestCase):

    def test_reindex_clusters(self):
        truth = np.array([1, 1, 2, 0, 0, 0])

        labels = np.array([0, 0, 1, 2, 2, 2])
        output = reindex_clusters(labels)
        self.assertTrue(np.array_equal(truth, output))

        labels = np.array([0, 0, 5, 2, 2, 2])
        output = reindex_clusters(labels, assume_range=False)
        self.assertTrue(np.array_equal(truth, output))
