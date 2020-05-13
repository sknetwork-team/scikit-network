#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for shortest_path.py"""
import unittest

from sknetwork.connectivity import distances, shortest_paths
from sknetwork.data import karate_club, cyclic_digraph


class TestShortestPath(unittest.TestCase):

    def test_parallel(self):
        adjacency = karate_club()
        dist1 = distances(adjacency)
        dist2 = distances(adjacency, n_jobs=-1)
        self.assertTrue((dist1 == dist2).all())

    def test_predecessors(self):
        adjacency = karate_club()
        _, predecessors = distances(adjacency, return_predecessors=True)
        self.assertTupleEqual(predecessors.shape, adjacency.shape)

    def test_shortest_paths(self):
        with self.assertRaises(ValueError):
            shortest_paths(cyclic_digraph(3), [0, 1], [0, 1])
