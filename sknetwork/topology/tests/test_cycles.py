#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for cycles.py"""
import unittest

import numpy as np
from scipy import sparse

from sknetwork.data import star_wars, house, cyclic_digraph, cyclic_graph, linear_digraph, linear_graph
from sknetwork.topology import is_connected, is_acyclic, get_cycles, break_cycles
from sknetwork.utils.format import bipartite2undirected, directed2undirected


class TestCycle(unittest.TestCase):

    def test_is_acyclic(self):
        adjacency_with_self_loops = sparse.identity(2, format='csr')
        self.assertFalse(is_acyclic(adjacency_with_self_loops))
        self.assertFalse(is_acyclic(adjacency_with_self_loops, directed=True))
        directed_cycle = cyclic_digraph(3)
        self.assertFalse(is_acyclic(directed_cycle))
        with self.assertRaises(ValueError):
            is_acyclic(directed_cycle, directed=False)
        undirected_line = linear_graph(2)
        self.assertTrue(is_acyclic(undirected_line))
        self.assertFalse(is_acyclic(undirected_line, directed=True))
        acyclic_graph = linear_digraph(2)
        self.assertTrue(is_acyclic(acyclic_graph))

    def test_get_cycles(self):
        adjacency_with_self_loops = sparse.identity(2, format='csr')
        node_cycles = get_cycles(adjacency_with_self_loops, directed=True)
        self.assertEqual(node_cycles, [[0], [1]])

        cycle_adjacency = cyclic_digraph(4)
        node_cycles = get_cycles(cycle_adjacency, directed=True)
        self.assertEqual(sorted(node_cycles[0]), [0, 1, 2, 3])
        adjacency_with_subcycles = cycle_adjacency + sparse.csr_matrix(([1], ([1], [3])), shape=cycle_adjacency.shape)
        node_cycles = get_cycles(adjacency_with_subcycles, directed=True)
        self.assertEqual(node_cycles, [[0, 1, 3], [0, 1, 2, 3]])

        undirected_cycle = cyclic_graph(4)
        node_cycles = get_cycles(undirected_cycle, directed=False)
        self.assertEqual(sorted(node_cycles[0]), [0, 1, 2, 3])

        disconnected_cycles = sparse.csr_matrix(([1, 1, 1], ([1, 2, 3], [2, 3, 1])), shape=(4, 4))
        node_cycles = get_cycles(disconnected_cycles, directed=True)
        self.assertEqual(sorted(node_cycles[0]), [1, 2, 3])

    def test_break_cycles(self):
        cycle_adjacency = cyclic_digraph(4)
        acyclic_graph = break_cycles(cycle_adjacency, root=0, directed=True)
        self.assertTrue(is_acyclic(acyclic_graph))
        adjacency_with_subcycles = cycle_adjacency + sparse.csr_matrix(([1], ([1], [0])), shape=cycle_adjacency.shape)
        acyclic_graph = break_cycles(adjacency_with_subcycles, root=0, directed=True)
        self.assertTrue(is_acyclic(acyclic_graph))

        undirected_cycle = house(metadata=False)
        acyclic_graph = break_cycles(undirected_cycle, root=0, directed=False)
        self.assertTrue(is_acyclic(acyclic_graph))

        disconnected_cycles = sparse.csr_matrix(([1, 1, 1, 1, 1], ([0, 1, 2, 3, 4], [1, 0, 3, 4, 2])), shape=(5, 5))
        self.assertFalse(is_connected(disconnected_cycles))
        acyclic_graph = break_cycles(disconnected_cycles, root=[0, 2], directed=True)
        self.assertTrue(is_acyclic(acyclic_graph))
