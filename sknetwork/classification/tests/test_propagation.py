#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for label propagation"""

import unittest

from sknetwork.classification import Propagation
from sknetwork.data import karate_club


class TestLabelPropagation(unittest.TestCase):

    def test_niter(self):
        adjacency = karate_club(metadata=False)
        n = adjacency.shape[0]
        seeds = {0: 0, 1: 1}
        propagation = Propagation(n_iter=3)
        labels = propagation.fit_transform(adjacency, seeds)
        self.assertEqual(len(labels), n)
