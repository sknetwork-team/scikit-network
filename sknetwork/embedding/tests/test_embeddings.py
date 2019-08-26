#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for embeddings"""

import unittest
from typing import Union

import numpy as np

from sknetwork.embedding import Spectral, SVD
from sknetwork.toy_graphs import random_graph, random_bipartite_graph, house


def barycenter_norm(adjacency, spectral: Spectral) -> np.ndarray:
    """Barycenter of the embedding with respect to adjacency weights.

    Parameters
    ----------
    adjacency
    spectral

    Returns
    -------
    barycenter

    """
    if spectral.normalized_laplacian:
        weights = adjacency.dot(np.ones(adjacency.shape[1]))
        if spectral.regularization:
            weights += spectral.regularization * adjacency.shape[1]
        return np.linalg.norm(spectral.embedding_.T.dot(weights)) / weights.sum()
    else:
        return np.linalg.norm(spectral.embedding_.mean(axis=0))


def has_proper_shape(adjacency, algo: Union[Spectral, SVD]) -> bool:
    """Check if the embedding has a proper shape with respect to the input data.

    Parameters
    ----------
    adjacency
    algo

    Returns
    -------

    """
    n, m = adjacency.shape
    k = algo.embedding_dimension
    if algo.embedding_.shape != (n, k):
        return False
    if algo.coembedding_ is not None and algo.coembedding_.shape != (m, k):
        return False
    if hasattr(algo, 'eigenvalues_') and len(algo.eigenvalues_) != k:
        return False
    if hasattr(algo, 'singular_values_') and len(algo.singular_values_) != k:
        return False

    return True


class TestEmbeddings(unittest.TestCase):

    def setUp(self):
        """

        Returns
        -------

        """
        self.adjacency = random_graph()
        self.biadjacency = random_bipartite_graph()
        self.house = house()

    def test_spectral_normalized(self):
        # Spectral with lanczos solver
        spectral = Spectral(2, normalized_laplacian=True, solver='lanczos')
        spectral.fit(self.adjacency)
        self.assertTrue(has_proper_shape(self.adjacency, spectral))
        self.assertTrue(min(spectral.eigenvalues_ >= -1e-6) and max(spectral.eigenvalues_ <= 2))

        spectral.fit(self.biadjacency)
        self.assertTrue(has_proper_shape(self.biadjacency, spectral))

        # test if the embedding is centered
        # without regularization
        spectral.regularization = None
        spectral.fit(self.house)
        self.assertAlmostEqual(barycenter_norm(self.house, spectral), 0)

        # with regularization
        spectral.regularization = 0.1
        spectral.fit(self.house)
        self.assertAlmostEqual(barycenter_norm(self.house, spectral), 0, places=2)

        # Spectral with halko solver
        spectral = Spectral(2, normalized_laplacian=True, solver='halko')
        spectral.fit(self.adjacency)
        self.assertTrue(has_proper_shape(self.adjacency, spectral))
        self.assertTrue(min(spectral.eigenvalues_ >= -1e-6) and max(spectral.eigenvalues_ <= 2))

        spectral.fit(self.biadjacency)
        self.assertTrue(has_proper_shape(self.biadjacency, spectral))
        self.assertTrue(type(spectral.eigenvalues_) == np.ndarray and len(spectral.eigenvalues_) == 2)

        # test if the embedding is centered
        # without regularization
        spectral.regularization = None
        spectral.fit(self.house)
        self.assertAlmostEqual(barycenter_norm(self.house, spectral), 0)

        # with regularization
        spectral.regularization = 0.1
        spectral.fit(self.house)
        self.assertAlmostEqual(barycenter_norm(self.house, spectral), 0, places=2)

    def test_spectral_basic(self):
        # Spectral with lanczos solver
        spectral = Spectral(2, normalized_laplacian=False, solver='lanczos')
        spectral.fit(self.adjacency)
        self.assertTrue(has_proper_shape(self.adjacency, spectral))
        self.assertTrue(min(spectral.eigenvalues_ >= -1) and max(spectral.eigenvalues_ <= 1))

        spectral.fit(self.biadjacency)
        self.assertTrue(has_proper_shape(self.biadjacency, spectral))

        # test if the embedding is centered
        # without regularization
        spectral.regularization = None
        spectral.fit(self.house)
        self.assertAlmostEqual(barycenter_norm(self.house, spectral), 0)

        # with regularization
        spectral.regularization = 0.1
        spectral.fit(self.house)
        self.assertAlmostEqual(barycenter_norm(self.house, spectral), 0)

    def test_svd(self):
        svd = SVD(2, solver='lanczos')
        svd.fit(self.adjacency)
        self.assertTrue(has_proper_shape(self.adjacency, svd))

        svd.fit(self.biadjacency)
        self.assertTrue(has_proper_shape(self.biadjacency, svd))

        svd = SVD(2, solver='halko')
        svd.fit(self.adjacency)
        self.assertTrue(has_proper_shape(self.adjacency, svd))

        svd.fit(self.biadjacency)
        self.assertTrue(has_proper_shape(self.biadjacency, svd))
