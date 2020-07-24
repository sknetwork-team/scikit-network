#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
import unittest

import numpy as np

from sknetwork.utils import KMeansDense


class TestKMeans(unittest.TestCase):

    def test_kmeans(self):
        x = np.random.randn(10, 3)
        kmeans = KMeansDense(n_clusters=2)
        labels = kmeans.fit_transform(x)
        self.assertEqual(labels.shape, (x.shape[0],))
        self.assertEqual(kmeans.cluster_centers_.shape, (kmeans.n_clusters, x.shape[1]))
