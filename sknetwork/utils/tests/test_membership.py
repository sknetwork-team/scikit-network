#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in July 2022
@author: Thomas Bonald <thomas.bonald@telecom-paris.fr>
"""
import unittest

import numpy as np

from sknetwork.utils.membership import get_membership, from_membership


class TestMembership(unittest.TestCase):

    def test_membership(self):
        labels = np.array([0, 0, 1, 2, 1, 1])
        membership = get_membership(labels)
        self.assertEqual(membership.nnz, 6)
        self.assertEqual(np.linalg.norm(labels - from_membership(membership)), 0)
        labels = np.array([0, 0, 1, 2, 1, -1])
        membership = get_membership(labels)
        self.assertEqual(membership.nnz, 5)
        self.assertEqual(np.linalg.norm(labels - from_membership(membership)), 0)
