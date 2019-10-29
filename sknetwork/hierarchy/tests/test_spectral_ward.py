#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

from scipy import sparse

from sknetwork.hierarchy import SpectralWard
from sknetwork.toy_graphs import karate_club


class TestSpectralClustering(unittest.TestCase):

    def setUp(self):
        self.undirected: sparse.csr_matrix = karate_club()

    def test_default_options(self):
        sw = SpectralWard(embedding_dimension=3)
        sw.fit(self.undirected)
        self.assertEqual(sw.dendrogram_.shape, (self.undirected.shape[0]-1, 4))
