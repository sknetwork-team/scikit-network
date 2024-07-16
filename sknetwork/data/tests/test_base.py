# -*- coding: utf-8 -*-
"""tests for dataset"""

import unittest

from sknetwork.data.base import Dataset


class TestDataset(unittest.TestCase):

    def test(self):
        dataset = Dataset(name='dataset')
        self.assertEqual(dataset.name, 'dataset')
        self.assertEqual(dataset['name'], 'dataset')
