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
    """Class for matrices with "sparse + low rank" structure.
    Example:

    :math:`A + xy^T`

    Parameters
    ----------
    sparse_mat: scipy.spmatrix
        Sparse component. Is converted to csr format automatically.
    low_rank_tuples: list
        List of tuple of arrays representing the low rank components [(x1, y1), (x2, y2),...].
        Each low rank component is of the form :math:`xy^T`.
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

    def _matvec(self, matrix: np.ndarray):
        """Right dot product with a dense matrix.

        Parameters
        ----------
        matrix:
            Matrix.

        Returns
        -------
        Dot product as a dense array
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
        SparseLR object
        """
        transposed_sparse = sparse.csr_matrix(self.sparse_mat.T)
        transposed_tuples = [(y, x) for (x, y) in self.low_rank_tuples]
        return SparseLR(transposed_sparse, transposed_tuples)

    def _adjoint(self):
        return self.transpose()

    def left_sparse_dot(self, matrix: sparse.csr_matrix):
        """Left dot product with a sparse matrix

        Parameters
        ----------
        matrix:
            Matrix

        Returns
        -------
        SparseLR object

        """
        return SparseLR(matrix.dot(self.sparse_mat), [(matrix.dot(x), y) for (x, y) in self.low_rank_tuples])

    def right_sparse_dot(self, matrix: sparse.csr_matrix):
        """Right dot product with a sparse matrix

        Parameters
        ----------
        matrix:
            Matrix

        Returns
        -------
        SparseLR object

        """
        return SparseLR(self.sparse_mat.dot(matrix), [(x, matrix.T.dot(y)) for (x, y) in self.low_rank_tuples])

    def astype(self, dtype: Union[str, np.dtype]):
        """Change dtype of the object.

        Parameters
        ----------
        dtype

        Returns
        -------
        SparseLR object

        """
        self.sparse_mat = self.sparse_mat.astype(dtype)
        self.low_rank_tuples = [(x.astype(dtype), y.astype(dtype)) for (x, y) in self.low_rank_tuples]
        if type(dtype) == np.dtype:
            self.dtype = dtype
        else:
            self.dtype = np.dtype(dtype)

        return self
