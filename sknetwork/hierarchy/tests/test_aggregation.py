#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

from sknetwork.hierarchy import Paris
from sknetwork.hierarchy import aggregate_dendrogram
from sknetwork.data import karate_club


class TestMetrics(unittest.TestCase):

    def test_aggregation(self):
        self.adjacency = karate_club()
        paris = Paris()
        self.dendrogram = paris.fit_transform(self.adjacency)

        n_clusters = 5
        dendrogram_, counts = aggregate_dendrogram(self.dendrogram, n_clusters, return_counts=True)
        self.assertEqual(dendrogram_.shape, (n_clusters - 1, self.dendrogram.shape[1]))
        self.assertEqual(counts.sum(), self.dendrogram.shape[0] + 1)
