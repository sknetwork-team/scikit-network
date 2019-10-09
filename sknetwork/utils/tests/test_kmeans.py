#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

import numpy as np

from sknetwork.utils import KMeans


class TestKMeans(unittest.TestCase):

    def setUp(self):
        self.x = np.random.randn(10, 3)

    def test_kmeans(self):
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(self.x)
        self.assertEqual(len(kmeans.labels_), self.x.shape[0])
        self.assertEqual(kmeans.cluster_centers_.shape, (kmeans.n_clusters, self.x.shape[1]))
