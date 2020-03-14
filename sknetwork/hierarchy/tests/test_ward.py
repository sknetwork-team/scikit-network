#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

from scipy import sparse

from sknetwork.hierarchy import Ward, BiWard
from sknetwork.data import karate_club, movie_actor


class TestWard(unittest.TestCase):

    def test_default_options(self):
        self.adjacency: sparse.csr_matrix = karate_club()
        spectral_ward = Ward(embedding_dimension=3)
        spectral_ward.fit(self.adjacency)
        self.assertEqual(spectral_ward.dendrogram_.shape, (self.adjacency.shape[0] - 1, 4))

    def test_bipartite(self):
        self.bipartite: sparse.csr_matrix = movie_actor(return_labels=False)
        bispectral_ward = BiWard(embedding_dimension=3, cluster_col=True)
        bispectral_ward.fit(self.bipartite)
        self.assertEqual(bispectral_ward.dendrogram_row_.shape, (self.bipartite.shape[0] - 1, 4))
        self.assertEqual(bispectral_ward.dendrogram_col_.shape, (self.bipartite.shape[1] - 1, 4))
