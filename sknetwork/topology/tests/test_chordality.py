#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Chordality testing"""
import unittest

from sknetwork.data import house, bow_tie, karate_club
from sknetwork.data.test_graphs import test_graph_empty, test_graph_clique
from sknetwork.topology.chordality import *


class TestChordality(unittest.TestCase):

    def test_empty(self):
        adjacency = test_graph_empty()
        self.assertTrue(is_chordal(adjacency))

    def test_cliques(self):
        adjacency = test_graph_clique()
        self.assertTrue(is_chordal(adjacency))

    def test_house(self):
        adjacency = house()
        self.assertFalse(is_chordal(adjacency))

    def test_bow_tie(self):
        adjacency = bow_tie()
        self.assertTrue(is_chordal(adjacency))

    def test_karate_club(self):
        adjacency = karate_club()
        self.assertFalse(is_chordal(adjacency))

