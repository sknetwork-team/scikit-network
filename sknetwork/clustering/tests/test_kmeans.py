#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest


from sknetwork.clustering import BiKMeans, KMeans
from sknetwork.data import karate_club, painters, movie_actor


class TestKMeans(unittest.TestCase):

    def test_undirected(self):
        adjacency = karate_club()
        kmeans = KMeans(n_clusters=2)
        labels = kmeans.fit_transform(adjacency)
        self.assertEqual(len(set(labels)), 2)

