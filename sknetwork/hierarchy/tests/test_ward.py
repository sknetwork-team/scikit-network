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
        dendrogram = Ward().fit_transform(self.adjacency)
        self.assertEqual(dendrogram.shape, (self.adjacency.shape[0] - 1, 4))

    def test_bipartite(self):
        self.bipartite: sparse.csr_matrix = movie_actor(return_labels=False)
        biward = BiWard(cluster_col=True, cluster_both=True)
        biward.fit(self.bipartite)
        n1, n2 = self.bipartite.shape
        self.assertEqual(biward.dendrogram_row_.shape, (n1 - 1, 4))
        self.assertEqual(biward.dendrogram_col_.shape, (n2 - 1, 4))
        self.assertEqual(biward.dendrogram_full_.shape, (n1 + n2 - 1, 4))
