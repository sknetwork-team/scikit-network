#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 19 2019
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""
from typing import Union, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator


class SparseLR(LinearOperator):
    """Class for matrices with "sparse + low rank" structure.
    Example:

    :math:`A + xy^T`

    Parameters
    ----------
    sparse_mat: scipy.spmatrix
        Sparse component. Is converted to csr format automatically.
    low_rank_tuples: list
        Single tuple of arrays of list of tuples, representing the low rank components [(x1, y1), (x2, y2),...].
        Each low rank component is of the form :math:`xy^T`.

    Examples
    --------
    >>> from scipy import sparse
    >>> from sknetwork.linalg import SparseLR
    >>> adjacency = sparse.eye(2, format='csr')
    >>> slr = SparseLR(adjacency, (np.ones(2), np.ones(2)))
    >>> x = np.ones(2)
    >>> slr.dot(x)
    array([3., 3.])
    >>> slr.sum(axis=0)
    array([3., 3.])
    >>> slr.sum(axis=1)
    array([3., 3.])
    >>> float(slr.sum())
    6.0

    References
    ----------
    De Lara (2019). `The Sparse + Low Rank trick for Matrix Factorization-Based Graph Algorithms.
    <http://www.mlgworkshop.org/2019/papers/MLG2019_paper_1.pdf>`_
    Proceedings of the 15th International Workshop on Mining and Learning with Graphs (MLG).
    """
    def __init__(self, sparse_mat: Union[sparse.csr_matrix, sparse.csc_matrix], low_rank_tuples: Union[list, Tuple],
                 dtype=float):
        n_row, n_col = sparse_mat.shape
        self.sparse_mat = sparse_mat.tocsr().astype(dtype)
        super(SparseLR, self).__init__(dtype=dtype, shape=(n_row, n_col))

        if isinstance(low_rank_tuples, Tuple):
            low_rank_tuples = [low_rank_tuples]
        self.low_rank_tuples = []
        for x, y in low_rank_tuples:
            if x.shape == (n_row,) and y.shape == (n_col,):
                self.low_rank_tuples.append((x.astype(self.dtype), y.astype(self.dtype)))
            else:
                raise ValueError('For each low rank tuple, x (resp. y) should be a vector of length {} (resp. {})'
                                 .format(n_row, n_col))

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
        """Transposed operator."""
        transposed_sparse = sparse.csr_matrix(self.sparse_mat.T)
        transposed_tuples = [(y, x) for (x, y) in self.low_rank_tuples]
        return SparseLR(transposed_sparse, transposed_tuples)

    def _adjoint(self):
        return self.transpose()

    def left_sparse_dot(self, matrix: sparse.csr_matrix):
        """Left dot product with a sparse matrix."""
        return SparseLR(matrix.dot(self.sparse_mat), [(matrix.dot(x), y) for (x, y) in self.low_rank_tuples])

    def right_sparse_dot(self, matrix: sparse.csr_matrix):
        """Right dot product with a sparse matrix."""
        return SparseLR(self.sparse_mat.dot(matrix), [(x, matrix.T.dot(y)) for (x, y) in self.low_rank_tuples])

    def sum(self, axis=None):
        """Row-wise, column-wise or total sum of operator's coefficients.

        Parameters
        ----------
        axis :
            If 0, return column-wise sum. If 1, return row-wise sum. Otherwise, return total sum.
        """
        if axis == 0:
            s = self.T.dot(np.ones(self.shape[0]))
        elif axis == 1:
            s = self.dot(np.ones(self.shape[1]))
        else:
            s = self.dot(np.ones(self.shape[1])).sum()
        return s

    def astype(self, dtype: Union[str, np.dtype]):
        """Change dtype of the object."""
        self.sparse_mat = self.sparse_mat.astype(dtype)
        self.low_rank_tuples = [(x.astype(dtype), y.astype(dtype)) for (x, y) in self.low_rank_tuples]
        self.dtype = np.dtype(dtype)

        return self
