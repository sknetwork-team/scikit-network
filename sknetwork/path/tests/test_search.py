#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for search.py"""

import unittest

import numpy as np

from sknetwork.data import cyclic_digraph
from sknetwork.data.test_graphs import *
from sknetwork.path import breadth_first_search


class TestSearch(unittest.TestCase):

    def test_bfs(self):
        adjacency = cyclic_digraph(3)
        search = breadth_first_search(adjacency, 0)
        search_ = np.arange(3)
        self.assertTrue(all(search == search_))

        adjacency = test_graph_empty()
        search = breadth_first_search(adjacency, 0)
        search_ = np.array([0])
        self.assertTrue(all(search == search_))

        adjacency = test_graph()
        search = breadth_first_search(adjacency, 3)
        search_ = np.array([3, 1, 2, 0, 4, 6, 8, 5, 7, 9])
        self.assertTrue(all(search == search_))

        adjacency = test_disconnected_graph()
        search = breadth_first_search(adjacency, 2)
        search_ = np.array([2, 3])
        self.assertTrue(all(search == search_))

        adjacency = test_digraph()
        search = breadth_first_search(adjacency, 1)
        search_ = {1, 3, 4, 2, 5}
        self.assertTrue(set(list(search)) == search_)
