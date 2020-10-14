#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for search.py"""

import unittest

import numpy as np

from sknetwork.data import cyclic_digraph
from sknetwork.path import breadth_first_search, depth_first_search


class TestSearch(unittest.TestCase):

    def setUp(self) -> None:
        """Load graph for tests."""
        self.adjacency = cyclic_digraph(3).astype(bool)

    def test_bfs(self):
        self.assertTrue((breadth_first_search(
            self.adjacency, 0, return_predecessors=False) == np.array([0, 1, 2])).all())

    def test_dfs(self):
        self.assertTrue((depth_first_search(
            self.adjacency, 0, return_predecessors=False) == np.array([0, 1, 2])).all())
