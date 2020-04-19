#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for d-iteration"""

import unittest

import numpy as np

from sknetwork.data import cyclic_digraph
from sknetwork.linalg.ppr_solver import pagerank


class TestDIter(unittest.TestCase):

    def test_cyclic(self):
        n = 5
        adjacency = cyclic_digraph(n)
        ref = np.ones(n) / n
        scores = pagerank(adjacency, ref)

        self.assertEqual(np.linalg.norm(ref - scores), 0)
