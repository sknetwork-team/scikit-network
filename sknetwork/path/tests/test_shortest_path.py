#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for shortest_path.py"""
import unittest

from sknetwork.data import karate_club, cyclic_digraph
# has to specify the exact file to avoid nosetests error on full tests
from sknetwork.path.shortest_path import get_distances, shortest_path


class TestShortestPath(unittest.TestCase):

    def test_parallel(self):
        adjacency = karate_club()
        dist1 = get_distances(adjacency)
        dist2 = get_distances(adjacency, n_jobs=-1)
        self.assertTrue((dist1 == dist2).all())

    def test_predecessors(self):
        adjacency = karate_club()
        _, predecessors = get_distances(adjacency, return_predecessors=True)
        self.assertTupleEqual(predecessors.shape, adjacency.shape)

    def test_shortest_paths(self):
        with self.assertRaises(ValueError):
            shortest_path(cyclic_digraph(3), [0, 1], [0, 1])

    def test_error_on_parallel_FW(self):
        adjacency = karate_club()
        self.assertRaises(ValueError, get_distances, adjacency, n_jobs=2, method='FW')

