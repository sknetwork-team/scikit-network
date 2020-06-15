#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for check.py"""

import unittest
import numpy as np

from sknetwork.utils.betweenness import betweenness_centrality


class TestChecks(unittest.TestCase):

    def setUp(self):
        pass

    def test_betweenness(self):
        # example from https://neo4j.com/docs/graph-algorithms/current/labs-algorithms/betweenness-centrality/
        M = np.array([
            [0,1,1,1,0,0], 
            [0,0,0,0,0,0], 
            [0,0,0,0,0,1], 
            [0,0,0,0,0,0], 
            [1,0,0,0,0,0], 
            [0,0,0,0,0,0]])

        ans = {2: 2, 0: 4}
        res = betweenness_centrality(M)
        self.assertDictEqual(res, ans)