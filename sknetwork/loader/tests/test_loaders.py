#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for loaders.py"""

import unittest
from os.path import isdir

from sknetwork.loader import load_vital_wikipedia, get_data_home


class TestLoader(unittest.TestCase):

    def test_wiki_vital(self):
        is_available = isdir(get_data_home() + '/vital_wikipedia')
        if is_available:
            adjacency = load_vital_wikipedia(outputs='adjacency')[0]
            self.assertEqual(adjacency.shape[0], 10012)
