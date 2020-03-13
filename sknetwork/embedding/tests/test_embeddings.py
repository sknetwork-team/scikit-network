#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for embeddings"""

import unittest
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.embedding import Spectral, BiSpectral
from sknetwork.data import karate_club, simple_bipartite_graph, house


def barycenter_norm(adjacency, spectral: Spectral) -> float:
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


def has_proper_shape(adjacency: Union[sparse.csr_matrix, np.ndarray], algorithm: Union[Spectral, BiSpectral]) -> bool:
    """Check if the embedding has a proper shape with respect to the input data.

    Parameters
    ----------
    adjacency: Adjacency matrix
    algorithm: Embedding

    Returns
    -------
    bool
    """
    n1, n2 = adjacency.shape
    k = algorithm.n_components
    if algorithm.embedding_.shape != (n1, k):
        return False
    if hasattr(algorithm, 'embedding_col_') and algorithm.embedding_col_.shape != (n2, k):
        return False
    if hasattr(algorithm, 'eigenvalues_') and len(algorithm.eigenvalues_) != k:
        return False
    if hasattr(algorithm, 'singular_values_') and len(algorithm.singular_values_) != k:
        return False

    return True


# noinspection DuplicatedCode,PyMissingOrEmptyDocstring
class TestEmbeddings(unittest.TestCase):

    def setUp(self):
        self.adjacency: sparse.csr_matrix = karate_club()
        self.house = house()

    def test_spectral_normalized(self):
        # Spectral with lanczos solver
        spectral = Spectral(2, normalized_laplacian=True, solver='lanczos')
        spectral.fit(self.adjacency)
        self.assertTrue(has_proper_shape(self.adjacency, spectral))
        self.assertTrue(min(spectral.eigenvalues_ >= -1e-6) and max(spectral.eigenvalues_ <= 2))

        # test if the embedding is centered
        # without regularization
        spectral.normalize = False
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

        # test if the embedding is centered
        # without regularization
        spectral.normalize = False
        spectral.regularization = None
        spectral.fit(self.adjacency)
        self.assertAlmostEqual(barycenter_norm(self.adjacency, spectral), 0)

        # with regularization
        spectral.regularization = 0.1
        spectral.fit(self.adjacency)
        self.assertAlmostEqual(barycenter_norm(self.adjacency, spectral), 0, places=2)

    def test_spectral_basic(self):
        # Spectral with lanczos solver
        spectral = Spectral(2, normalized_laplacian=False, barycenter=False, normalize=False, solver='lanczos')
        spectral.fit(self.adjacency)
        self.assertTrue(has_proper_shape(self.adjacency, spectral))
        self.assertTrue(min(spectral.eigenvalues_ >= 0) and max(spectral.eigenvalues_ <= 1))

        # test if the embedding is centered
        # without regularization
        spectral.fit(self.house)
        self.assertAlmostEqual(barycenter_norm(self.house, spectral), 0)

        # with regularization
        spectral.regularization = 0.1
        spectral.fit(self.house)
        self.assertAlmostEqual(barycenter_norm(self.house, spectral), 0)

    def test_spectral_equalize(self):
        spectral = Spectral(2, equalize=True, barycenter=False, normalize=False)
        spectral.regularization = None
        spectral.fit(self.house)
        self.assertAlmostEqual(barycenter_norm(self.house, spectral), 0)

    def test_predict(self):
        spectral = Spectral(4)
        spectral.fit(self.adjacency)
        # single node
        error = np.sum(np.abs(spectral.predict(self.adjacency[0]) - spectral.embedding_[0]))
        self.assertAlmostEqual(error, 0)
        # all nodes
        error = np.sum(np.abs(spectral.predict(self.adjacency) - spectral.embedding_))
        self.assertAlmostEqual(error, 0)

    def test_bispectral(self):
        self.biadjacency = simple_bipartite_graph()
        bispectral = BiSpectral(solver='lanczos')
        bispectral.fit(self.biadjacency)
        self.assertTrue(has_proper_shape(self.biadjacency, bispectral))

        bispectral = BiSpectral(2, solver='halko')
        bispectral.fit(self.biadjacency)
        self.assertTrue(has_proper_shape(self.biadjacency, bispectral))

        bispectral = BiSpectral(2, normalized_laplacian=False)
        bispectral.fit(self.biadjacency)
        self.assertTrue(has_proper_shape(self.biadjacency, bispectral))

    def test_bispectral_predict(self):
        self.biadjacency = simple_bipartite_graph()
        bispectral = BiSpectral(3)
        bispectral.fit(self.biadjacency)
        # single node
        error = np.sum(np.abs(bispectral.predict(self.biadjacency[0]) - bispectral.embedding_[0]))
        self.assertAlmostEqual(error, 0)
        # all nodes
        error = np.sum(np.abs(bispectral.predict(self.biadjacency) - bispectral.embedding_))
        self.assertAlmostEqual(error, 0)
