#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 19 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from typing import Union


class SparseLR(LinearOperator):
    """Class for matrices with "sparse + low rank" structure. Ex: A + xy^T.

    Parameters
    ----------
    sparse_mat: scipy.spmatrix
        sparse component. Is converted to csr format automatically.
    low_rank_tuples: list
        list of tuple of arrays representing the low rank components [(x1, y1), (x2, y2),...].
        Each low rank component is of the form xy^T.

    """

    def __init__(self, sparse_mat: Union[sparse.csr_matrix, sparse.csc_matrix], low_rank_tuples: list):
        self.sparse_mat = sparse_mat.tocsr()
        self.low_rank_tuples = []
        LinearOperator.__init__(self, self.sparse_mat.dtype, self.sparse_mat.shape)
        for x, y in low_rank_tuples:
            if x.shape == (self.shape[0],) and y.shape == (self.shape[1],):
                self.low_rank_tuples.append((x.astype(self.dtype), y.astype(self.dtype)))
            else:
                raise ValueError(
                    'For each low rank tuple, x (resp. y) should be a vector of lenght n_rows (resp. n_cols)')

    def __neg__(self):
        return SparseLR(-self.sparse_mat, [(-x, y) for (x, y) in self.low_rank_tuples])

    def __add__(self, other: 'SparseLR'):
        if type(other) == sparse.csr_matrix:
            return SparseLR(self.sparse_mat + other, self.low_rank_tuples)
        else:
            return SparseLR(self.sparse_mat + other.sparse_mat, self.low_rank_tuples + other.low_rank_tuples)

    def __sub__(self, other):
        return self.__add__(-other)

    def __mul__(self, other):
        return SparseLR(other * self.sparse_mat, [(other * x, y) for (x, y) in self.low_rank_tuples])

    def _matvec(self, matrix):
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

    def _transpose(self):
        """Transposed matrix.

        Returns
        -------
        a SparseLR object
        """
        transposed_sparse = sparse.csr_matrix(self.sparse_mat.T)
        transposed_tuples = [(y, x) for (x, y) in self.low_rank_tuples]
        return SparseLR(transposed_sparse, transposed_tuples)

    def left_sparse_dot(self, matrix):
        """Left dot product with a sparse matrix

        Parameters
        ----------
        matrix: sparse.csr_matrix

        Returns
        -------
        a SparseLR object

        """
        return SparseLR(matrix.dot(self.sparse_mat), [(matrix.dot(x), y) for (x, y) in self.low_rank_tuples])

    def right_sparse_dot(self, matrix):
        """Right dot product with a sparse matrix

        Parameters
        ----------
        matrix: sparse.csr_matrix

        Returns
        -------
        a SparseLR object

        """
        return SparseLR(self.sparse_mat.dot(matrix), [(x, matrix.T.dot(y)) for (x, y) in self.low_rank_tuples])

    def astype(self, dtype):
        """Change dtype of the object.

        Parameters
        ----------
        dtype

        Returns
        -------
        a SparseLR object

        """
        self.sparse_mat = self.sparse_mat.astype(dtype)
        self.low_rank_tuples = [(x.astype(dtype), y.astype(dtype)) for (x, y) in self.low_rank_tuples]
        self.dtype = dtype

        return self
