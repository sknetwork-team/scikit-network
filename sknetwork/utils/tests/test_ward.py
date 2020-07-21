#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
import unittest

import numpy as np

from sknetwork.utils import WardDense


class TestKMeans(unittest.TestCase):

    def test_kmeans(self):
        x = np.random.randn(10, 3)
        ward = WardDense()
        dendrogram = ward.fit_transform(x)
        self.assertEqual(dendrogram.shape, (x.shape[0] - 1, 4))
