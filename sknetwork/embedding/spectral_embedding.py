#!/usr/bin/env python3
# coding: utf-8

"""
Created on Thu Aug 2 2018

@author: Nathan de Lara <ndelara@enst.fr>

Spectral embedding by decomposition of the normalized graph Laplacian.
"""

import networkx as nx
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import eigsh


class SpectralEmbedding:
    """Spectral embeddings for non-linear dimensionality reduction.
        .. math::
            X = (\Lambda^{1/2})^{\dagger}V^TD^{-1/2}
        where 'V\Lambda V^T' is the decomposition of the normalized graph Laplacian.
        Parameters
        -----------
        n_components : integer, default: 2
            The dimension of the projected subspace.

        Attributes
        ----------
        embedding_ : array, shape = (n_samples, n_components)
            Embedding matrix of the nodes.

        eigenvalues_ : array, shape = (n_components)
            Smallest eigenvalues of the training matrix

        References
        ----------
        -
        """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.embedding_ = None
        self.eigenvalues_ = None

    def fit(self, graph, tol=1e-6):
        """Fits the model from data in adjacency_matrix.
        Parameters
        ----------
        graph: networkx graph, Scipy scr matrix or numpy ndarray.

        tol: float, default: 1e-6
            Tolerance for pseudo inverse computation of eigen values.
        """
        if type(graph) == sparse.csr_matrix:
            adj_matrix = graph
        elif type(graph) == np.ndarray:
            adj_matrix = sparse.csr_matrix(graph)
        elif type(graph) == nx.classes.graph.Graph:
            adj_matrix = nx.adjacency_matrix(graph)
        elif type(graph) == nx.classes.digraph.DiGraph:
            adj_matrix = nx.adjacency_matrix(graph.to_undirected())
        else:
            raise TypeError(
                "The argument should be a NetworkX graph, a NumPy array or a SciPy Compressed Sparse Row matrix.")
        n_nodes, m_nodes = adj_matrix.shape
        if n_nodes != m_nodes:
            raise ValueError("The adjacency matrix is not square.")

        diags = adj_matrix.sum(axis=1).flatten()
        degree_matrix = sparse.spdiags(diags, [0], n_nodes, n_nodes, format='csr')
        laplacian = degree_matrix - adj_matrix
        with scipy.errstate(divide='ignore'):
            diags_sqrt = 1.0 / scipy.sqrt(diags)
        diags_sqrt[scipy.isinf(diags_sqrt)] = 0
        dh = sparse.spdiags(diags_sqrt, [0], n_nodes, n_nodes, format='csr')
        laplacian = dh.dot(laplacian.dot(dh))

        w, v = eigsh(laplacian, self.n_components, which='SM')

        self.eigenvalues_ = w
        self.embedding_ = np.array([1 / np.sqrt(x) if x > tol else 0 for x in w]) * v.dot(dh)

        return self
