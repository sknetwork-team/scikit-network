#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

from scipy import sparse

from sknetwork.clustering import SpectralClustering
from sknetwork.embedding import BiSpectral
from sknetwork.toy_graphs import karate_club, painters, movie_actor
from sknetwork.utils import KMeans


class TestSpectralClustering(unittest.TestCase):

    def setUp(self):
        self.undirected: sparse.csr_matrix = karate_club()

    def test_default_options(self):
        sc = SpectralClustering(embedding_dimension=3)
        sc.fit(self.undirected)
        self.assertEqual(sc.labels_.shape[0], self.undirected.shape[0])

