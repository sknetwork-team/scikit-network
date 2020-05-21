#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for shortest_path.py"""
import unittest

from sknetwork.path import distance, shortest_path
from sknetwork.data import karate_club, cyclic_digraph


class TestShortestPath(unittest.TestCase):

    def test_parallel(self):
        adjacency = karate_club()
        dist1 = distance(adjacency)
        dist2 = distance(adjacency, n_jobs=-1)
        self.assertTrue((dist1 == dist2).all())

    def test_predecessors(self):
        adjacency = karate_club()
        _, predecessors = distance(adjacency, return_predecessors=True)
        self.assertTupleEqual(predecessors.shape, adjacency.shape)

    def test_shortest_paths(self):
        with self.assertRaises(ValueError):
            shortest_path(cyclic_digraph(3), [0, 1], [0, 1])
