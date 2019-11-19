#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

from scipy import sparse

from sknetwork.hierarchy import SpectralWard, BiSpectralWard
from sknetwork.toy_graphs import karate_club, movie_actor


class TestSpectralClustering(unittest.TestCase):

    def test_default_options(self):
        self.undirected: sparse.csr_matrix = karate_club()
        sw = SpectralWard(embedding_dimension=3)
        sw.fit(self.undirected)
        self.assertEqual(sw.dendrogram_.shape, (self.undirected.shape[0]-1, 4))

    def test_bipartite(self):
        self.bipartite: sparse.csr_matrix = movie_actor(return_labels=False)
        bsw = BiSpectralWard(embedding_dimension=3)
        bsw.fit(self.bipartite)
        self.assertEqual(bsw.dendrogram_.shape, (self.bipartite.shape[0] - 1, 4))
        self.assertEqual(bsw.col_dendrogram_.shape, (self.bipartite.shape[1] - 1, 4))
