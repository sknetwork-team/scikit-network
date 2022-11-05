#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for clustering post-processing"""
import unittest

import numpy as np

from sknetwork.data import house, star_wars
from sknetwork.clustering.postprocess import reindex_labels, aggregate_graph


class TestClusteringPostProcessing(unittest.TestCase):

    def test_reindex_clusters(self):
        truth = np.array([1, 1, 2, 0, 0, 0])

        labels = np.array([0, 0, 1, 2, 2, 2])
        output = reindex_labels(labels)
        self.assertTrue(np.array_equal(truth, output))

        labels = np.array([0, 0, 5, 2, 2, 2])
        output = reindex_labels(labels)
        self.assertTrue(np.array_equal(truth, output))

    def test_aggregate_graph(self):
        adjacency = house()
        labels = np.array([0, 0, 1, 1, 2])
        aggregate = aggregate_graph(adjacency, labels)
        self.assertEqual(aggregate.shape, (3, 3))

        biadjacency = star_wars()
        labels = np.array([0, 0, 1, 2])
        labels_row = np.array([0, 1, 3, -1])
        labels_col = np.array([0, 0, 1])
        aggregate = aggregate_graph(biadjacency, labels=labels, labels_col=labels_col)
        self.assertEqual(aggregate.shape, (3, 2))
        self.assertEqual(aggregate.shape, (3, 2))
        aggregate = aggregate_graph(biadjacency, labels_row=labels_row, labels_col=labels_col)
        self.assertEqual(aggregate.shape, (4, 2))
