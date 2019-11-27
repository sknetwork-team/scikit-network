#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for search.py"""

import unittest

import numpy as np

from sknetwork.basics import breadth_first_search, depth_first_search
from sknetwork.data import rock_paper_scissors


class TestSearch(unittest.TestCase):

    def setUp(self) -> None:
        self.rock_paper_scissors = rock_paper_scissors()

    def test_bfs(self):
        self.assertTrue((breadth_first_search(
            self.rock_paper_scissors, 0, return_predecessors=False) == np.array([0, 1, 2])).all())

    def test_dfs(self):
        self.assertTrue((depth_first_search(
            self.rock_paper_scissors, 0, return_predecessors=False) == np.array([0, 1, 2])).all())

