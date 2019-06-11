#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 19 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import numpy as np
from scipy import sparse
from typing import Union


class SparseLR:
    """Class for matrices with "sparse + low rank" structure. Ex: A + xy^T.

    Parameters
    ----------
    sparse_mat: scipy.spmatrix
        sparse component. Is converted to csr format automatically.
    low_rank_tuples: list
        list of tuple of arrays representing the low rank components [(x1, y1), (x2, y2),...].
        Each low rank component is of the form xy^T.

    Attributes
    ----------
    shape: tuple
        shape of the matrix, same as sparse_mat.shape
    dtype: data type
        same as sparse_mat.dtype
    """

    def __init__(self, sparse_mat: Union[sparse.csr_matrix, sparse.csc_matrix], low_rank_tuples: list):
        self.sparse_mat = sparse_mat.tocsr()
        self.low_rank_tuples = []
        self.shape = self.sparse_mat.shape
        self.dtype = self.sparse_mat.dtype
        for x, y in low_rank_tuples:
            if x.shape == (self.shape[0],) and y.shape == (self.shape[1],):
                self.low_rank_tuples.append((x.astype(self.dtype), y.astype(self.dtype)))
            else:
                raise ValueError(
                    'For each low rank tuple, x (resp. y) should be a vector of lenght n_rows (resp. n_cols)')

    def __add__(self, other: 'SparseLR'):
        return SparseLR(self.sparse_mat + other.sparse_mat, self.low_rank_tuples + other.low_rank_tuples)

    def dot(self, matrix) -> np.ndarray:
        """Right dot product with a dense matrix.

        Parameters
        ----------
        matrix: np.ndarray

        Returns
        -------
        the dot product as a dense array
        """
        prod = self.sparse_mat.dot(matrix)
        if len(matrix.shape) == 1:
            for (x, y) in self.low_rank_tuples:
                prod += x * matrix.dot(y)
        else:
            transposed = matrix.T
            for (x, y) in self.low_rank_tuples:
                prod += x[:, np.newaxis].dot(transposed.dot(y)[:, np.newaxis].T)
        return prod

    def transpose(self):
        """Transposed matrix.

        Returns
        -------
        a SparseLR object
        """
        transposed_sparse = sparse.csr_matrix(self.sparse_mat.T)
        transposed_tuples = [(y, x) for (x, y) in self.low_rank_tuples]
        return SparseLR(transposed_sparse, transposed_tuples)

    # noinspection PyPep8Naming
    @property
    def T(self):
        return self.transpose()
