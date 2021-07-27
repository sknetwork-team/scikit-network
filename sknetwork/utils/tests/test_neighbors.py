#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 2020
@author: Thomas Bonald <bonald@enst.fr>
"""
import unittest

import numpy as np
from numpy.linalg import norm

from sknetwork.data import karate_club
from sknetwork.utils import get_neighbors


class TestNeighbors(unittest.TestCase):

    def test_graph(self):
        adjacency = karate_club()
        neighbors = get_neighbors(adjacency, 5)
        neighbors_true = np.array([0, 6, 10, 16])
        self.assertEqual(norm(neighbors - neighbors_true), 0)
