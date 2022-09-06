#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 2020
@author: Thomas Bonald <bonald@enst.fr>
"""
import unittest

import numpy as np
from numpy.linalg import norm

from sknetwork.data import karate_club, painters
from sknetwork.utils import get_neighbors, get_degrees, get_weights


class TestNeighbors(unittest.TestCase):

    def test_graph(self):
        adjacency = karate_club()
        neighbors = get_neighbors(adjacency, 5)
        degrees = get_degrees(adjacency)
        neighbors_true = np.array([0, 6, 10, 16])
        self.assertEqual(norm(neighbors - neighbors_true), 0)
        self.assertEqual(degrees[5], 4)

    def test_digraph(self):
        adjacency = painters()
        neighbors = get_neighbors(adjacency, 0)
        out_degrees = get_degrees(adjacency)
        out_weights = get_weights(adjacency)
        neighbors_true = np.array([3, 10])
        self.assertEqual(norm(neighbors - neighbors_true), 0)
        self.assertEqual(out_degrees[0], 2)
        self.assertEqual(out_weights[0], 2)
        neighbors = get_neighbors(adjacency, 0, transpose=True)
        in_degrees = get_degrees(adjacency, transpose=True)
        in_weights = get_weights(adjacency, transpose=True)
        neighbors_true = np.array([3,  6,  8, 10, 11])
        self.assertEqual(norm(neighbors - neighbors_true), 0)
        self.assertEqual(in_degrees[0], 5)
        self.assertEqual(in_weights[0], 5)
