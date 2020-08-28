#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for dag.py"""
import unittest

import numpy as np

from sknetwork.data import house
from sknetwork.topology import DAG


class TestDAG(unittest.TestCase):

    def test_options(self):
        adjacency = house()
        dag = DAG()
        dag.fit(adjacency)
        self.assertEqual(dag.indptr_.shape[0], adjacency.shape[0]+1)
        self.assertEqual(dag.indices_.shape[0], 6)

        with self.assertRaises(ValueError):
            dag.fit(adjacency, sorted_nodes=np.arange(3))

        with self.assertRaises(ValueError):
            dag = DAG(ordering='toto')
            dag.fit(adjacency)
