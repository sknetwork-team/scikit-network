#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for search.py"""

import unittest

import numpy as np

from sknetwork.data import cyclic_digraph
from sknetwork.data.test_graphs import *
from sknetwork.path import get_dag


class TestSearch(unittest.TestCase):

    def test(self):
        adjacency = cyclic_digraph(3)
        dag = get_dag(adjacency)
        self.assertEqual(dag.nnz, 2)

        adjacency = test_graph_empty()
        dag = get_dag(adjacency)
        self.assertEqual(dag.nnz, 0)

        adjacency = test_graph()
        dag = get_dag(adjacency)
        self.assertEqual(dag.nnz, 12)
        dag = get_dag(adjacency, order=np.arange(10) % 3)
        self.assertEqual(dag.nnz, 10)

        adjacency = test_disconnected_graph()
        dag = get_dag(adjacency, 3)
        self.assertEqual(dag.nnz, 1)

        adjacency = test_digraph()
        dag = get_dag(adjacency, 1)
        self.assertEqual(dag.nnz, 4)
