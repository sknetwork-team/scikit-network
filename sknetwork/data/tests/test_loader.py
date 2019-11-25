#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for loading.py"""

import unittest
import tempfile

from sknetwork.data import load_vital_wikipedia, get_data_home, clear_data_home


class TestLoader(unittest.TestCase):

    def test_wiki_vital(self):
        tmp_data_dir = tempfile.gettempdir() + '/vital_wikipedia'
        clear_data_home(tmp_data_dir)
        adjacency, _, _, _, _ = \
            load_vital_wikipedia(return_labels=True, return_labels_true=True)
        self.assertEqual(adjacency.shape[0], 10012)
        clear_data_home(tmp_data_dir)
