#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for loading.py"""

import unittest
import tempfile

from sknetwork.data import load_vital_wikipedia, clear_data_home


class TestLoader(unittest.TestCase):

    def test_wiki_vital(self):
        tmp_data_dir = tempfile.gettempdir() + '/vital_wikipedia'
        clear_data_home(tmp_data_dir)
        adjacency, _, _, _, _ = \
            load_vital_wikipedia(return_labels=True, return_categories=True)
        adjacency_bis, _ = load_vital_wikipedia(outputs='adjacency', return_titles=True)
        adjacency_ter = load_vital_wikipedia(outputs='adjacency')
        self.assertTrue((adjacency.data == adjacency_bis.data).all())
        self.assertTrue((adjacency.data == adjacency_ter.data).all())
        clear_data_home(tmp_data_dir)
