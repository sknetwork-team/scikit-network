#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 12 2018
@author: Nathan de Lara <ndelara@enst.fr>
Part of this code was adapted from the scikit-learn project: https://scikit-learn.org/stable/.
"""

import numpy as np
from scipy import sparse, linalg
from sknetwork.utils.sparse_lowrank import SparseLR
from sknetwork.utils.checks import check_random_state
from typing import Union


def safe_sparse_dot(a, b, dense_output=False):
    """
    Dot product that handle the sparse matrix case correctly
    Uses BLAS GEMM as replacement for numpy.dot where possible
    to avoid unnecessary copies.
    Parameters
    ----------
    a : array or sparse matrix
    b : array or sparse matrix
    dense_output : boolean, default False
        When False, either ``a`` or ``b`` being sparse will yield sparse
        output. When True, output will always be an array.
    Returns
    -------
    dot_product : array or sparse matrix
        sparse if ``a`` or ``b`` is sparse and ``dense_output=False``.
    """
    if type(a) == SparseLR and type(b) == np.ndarray:
        return a.dot(b)
    if type(b) == SparseLR and type(a) == np.ndarray:
        return b.T.dot(a.T).T
    if type(a) == SparseLR and type(b) == SparseLR:
        raise NotImplementedError
    if type(a) == SparseLR and type(b) == sparse.csr_matrix:
        return a.right_sparse_dot(b)
    if type(b) == SparseLR and type(a) == sparse.csr_matrix:
        return b.left_sparse_dot(a)
    if sparse.issparse(a) or sparse.issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)


def randomized_range_finder(matrix, size, n_iter, power_iteration_normalizer='auto', random_state=None,
                            return_all: bool = False) -> [np.ndarray, tuple]:
    """Computes an orthonormal matrix whose range approximates the range of A.

    Parameters
    ----------
    matrix : two dimensional array
        The input data matrix
    size : integer
        Size of the return array
    n_iter : int
        Number of power iterations. It can be used to deal with very noisy
        problems. When 'auto', it is set to 4, unless ``size`` is small
        (< .1 * min(matrix.shape)) in which case ``n_iter`` is set to 7.
        This improves precision with few components.
    power_iteration_normalizer: ``'auto'`` (default), ``'QR'``, ``'LU'``, ``None``
            Whether the power iterations are normalized with step-by-step
            QR factorization (the slowest but most accurate), ``None``
            (the fastest but numerically unstable when ``n_iter`` is large, e.g.
            typically 5 or larger), or ``'LU'`` factorization (numerically stable
            but can lose slightly in accuracy). The ``'auto'`` mode applies no
            normalization if ``n_iter`` <= 2 and switches to ``'LU'`` otherwise.
    random_state: int, RandomState instance or ``None``, optional (default= ``None``)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If ``None``, the random number generator is the RandomState
        instance used by `np.random`.
    return_all : if True, returns (range_matrix, random_matrix, random_proj)
                else returns range_matrix.

    Returns
    -------
    range_matrix : 2D array
        matrix (size x size) projection matrix, the range of which
        approximates well the range of the input matrix.

    Notes
    -----
    Follows Algorithm 4.3 of

    Finding structure with randomness: Stochastic algorithms for constructing approximate matrix decompositions
    Halko, et al., 2009 (arXiv:909) http://arxiv.org/pdf/0909.4061
    """
    random_state = check_random_state(random_state)

    # Generating normal random vectors with shape: (A.shape[1], size)
    random_matrix = random_state.normal(size=(matrix.shape[1], size))
    if matrix.dtype.kind == 'f':
        # Ensure f32 is preserved as f32
        random_matrix = random_matrix.astype(matrix.dtype, copy=False)
    range_matrix = random_matrix.copy()

    # Deal with "auto" mode
    if power_iteration_normalizer == 'auto':
        if n_iter <= 2:
            power_iteration_normalizer = 'none'
        else:
            power_iteration_normalizer = 'LU'

    # Perform power iterations with 'range_matrix' to further 'imprint' the top
    # singular vectors of matrix in 'range_matrix'
    for i in range(n_iter):
        if power_iteration_normalizer == 'none':
            range_matrix = safe_sparse_dot(matrix, range_matrix)
            range_matrix = safe_sparse_dot(matrix.T, range_matrix)
        elif power_iteration_normalizer == 'LU':
            range_matrix, _ = linalg.lu(safe_sparse_dot(matrix, range_matrix), permute_l=True)
            range_matrix, _ = linalg.lu(safe_sparse_dot(matrix.T, range_matrix), permute_l=True)
        elif power_iteration_normalizer == 'QR':
            range_matrix, _ = linalg.qr(safe_sparse_dot(matrix, range_matrix), mode='economic')
            range_matrix, _ = linalg.qr(safe_sparse_dot(matrix.T, range_matrix), mode='economic')

    # Sample the range of 'matrix' using by linear projection of 'range_matrix'
    # Extract an orthonormal basis
    range_matrix, _ = linalg.qr(safe_sparse_dot(matrix, range_matrix), mode='economic')
    if return_all:
        return range_matrix, random_matrix, matrix.dot(random_matrix)
    else:
        return range_matrix


# =====================
#          SVD
# =====================


def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.
    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    Parameters
    ----------
    u, v: np.ndarray
        u and v are the output of the svd, with matching inner dimensions so one can compute `np.dot(u * s, v)`.
    u_based_decision : bool, (default=True)
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v.
        The choice of which variable to base the decision on is generally algorithm dependent.

    Returns
    -------
    u_adjusted, v_adjusted: arrays with the same dimensions as the input.
    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]

    return u, v


def randomized_svd(matrix, n_components: int, n_oversamples: int = 10, n_iter='auto', transpose='auto',
                   power_iteration_normalizer: Union[str, None] = 'auto', flip_sign: bool = True, random_state=None):
    """Computes a truncated randomized SVD

        Parameters
        ----------
        matrix : ndarray or sparse matrix
            Matrix to decompose
        n_components : int
            Number of singular values and vectors to extract.
        n_oversamples : int (default=10)
            Additional number of random vectors to sample the range of M so as
            to ensure proper conditioning. The total number of random vectors
            used to find the range of M is embedding_dimension + n_oversamples. Smaller
            number can improve speed but can negatively impact the quality of
            approximation of singular vectors and singular values.
        n_iter : int or 'auto' (default is 'auto')
            See :meth:`randomized_range_finder`
        power_iteration_normalizer : ``'auto'`` (default), ``'QR'``, ``'LU'``, ``None``
            See :meth:`randomized_range_finder`
        transpose : True, False or 'auto' (default)
            Whether the algorithm should be applied to ``matrix.T`` instead of ``matrix``. The
            result should approximately be the same. The 'auto' mode will
            trigger the transposition if ``matrix.shape[1] > matrix.shape[0]`` since this
            implementation of randomized SVD tends to be a little faster in that case.
        flip_sign : boolean, (default=True)
            The output of a singular value decomposition is only unique up to a
            permutation of the signs of the singular vectors. If `flip_sign` is
            set to `True`, the sign ambiguity is resolved by making the largest
            loadings for each component in the left singular vectors positive.
        random_state : int, RandomState instance or None, optional (default=None)
            See :meth:`randomized_range_finder`

        Returns
        -------
        left_singular_vectors: np.ndarray
        singular_values: np.ndarray
        right_singular_vectors: np.ndarray

        Notes
        -----
        This algorithm finds a (usually very good) approximate truncated
        singular value decomposition using randomization to speed up the
        computations. It is particularly fast on large matrices on which
        you wish to extract only a small number of components. In order to
        obtain further speed up, ``n_iter`` can be set <=2 (at the cost of
        loss of precision).

        References
        ----------
        * Finding structure with randomness: Stochastic algorithms for constructing
          approximate matrix decompositions
          Halko, et al., 2009 http://arxiv.org/abs/arXiv:0909.4061
          (algorithm 5.1)
        * A randomized algorithm for the decomposition of matrices
          Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert
        * An implementation of a randomized algorithm for principal component
          analysis
          A. Szlam et al. 2014
        """

    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples
    n_samples, n_features = matrix.shape

    if n_iter == 'auto':
        # Checks if the number of iterations is explicitly specified
        # Adjust n_iter. 7 was found a good compromise for PCA. See #5299
        n_iter = 7 if n_components < .1 * min(matrix.shape) else 4

    if transpose == 'auto':
        transpose = n_samples < n_features
    if transpose:
        # this implementation is a bit faster with smaller shape[1]
        matrix = matrix.T

    range_matrix: np.ndarray = randomized_range_finder(matrix, n_random, n_iter,
                                                       power_iteration_normalizer, random_state)

    # project M to the (k + p) dimensional space using the basis vectors
    approx_matrix = safe_sparse_dot(range_matrix.T, matrix)

    # compute the SVD on the thin matrix: (k + p) wide
    uhat, singular_values, v = linalg.svd(approx_matrix, full_matrices=False)

    del approx_matrix
    u = np.dot(range_matrix, uhat)

    if flip_sign:
        if not transpose:
            u, v = svd_flip(u, v)
        else:
            # In case of transpose u_based_decision=false
            # to actually flip based on u and not v.
            u, v = svd_flip(u, v, u_based_decision=False)

    if transpose:
        # transpose back the results according to the input convention
        return v[:n_components, :].T, singular_values[:n_components], u[:, :n_components].T
    else:
        return u[:, :n_components], singular_values[:n_components], v[:n_components, :]


# =============================
# Eigenvalue Decomposition
# =============================


def randomized_eig(matrix, n_components: int, which='LM', n_oversamples: int = 10, n_iter='auto',
                   power_iteration_normalizer: Union[str, None] = 'auto', random_state=None, one_pass: bool = False):
    """Computes a randomized eigenvalue decomposition.

    Parameters
    ----------
    matrix: ndarray or sparse matrix
        Matrix to decompose
    n_components: int
        Number of singular values and vectors to extract.
    which: str
        which eigenvalues to compute. ``'LM'`` for Largest Magnitude and ``'SM'`` for Smallest Magnitude.
        Any other entry will result in Largest Magnitude.
    n_oversamples : int (default=10)
        Additional number of random vectors to sample the range of ``matrix`` so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of ``matrix`` is ``n_components + n_oversamples``. Smaller number can improve speed
        but can negatively impact the quality of approximation of singular vectors and singular values.
    n_iter: int or 'auto' (default is 'auto')
        See :meth:`randomized_range_finder`
    power_iteration_normalizer: ``'auto'`` (default), ``'QR'``, ``'LU'``, ``None``
        See :meth:`randomized_range_finder`
    random_state: int, RandomState instance or None, optional (default=None)
        See :meth:`randomized_range_finder`
    one_pass: bool (default=False)
        whether to use algorithm 5.6 instead of 5.3. 5.6 requires less access to the original matrix,
        while 5.3 is more accurate.

    Returns
    -------
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    References
    ----------
    Finding structure with randomness: Stochastic algorithms for constructing
    approximate matrix decompositions
    Halko, et al., 2009
    http://arxiv.org/abs/arXiv:0909.4061
    """

    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples
    n_samples, n_features = matrix.shape
    lambda_max = None

    if n_samples != n_features:
        raise ValueError('The input matrix is not square.')

    if which == 'SM':
        lambda_max, _ = randomized_eig(matrix, n_components=1)
        matrix *= -1
        if isinstance(matrix, SparseLR):
            matrix += SparseLR(lambda_max[0] * sparse.identity(matrix.shape[0]), [])
        else:
            matrix += lambda_max[0] * sparse.identity(matrix.shape[0])

    if n_iter == 'auto':
        # Checks if the number of iterations is explicitly specified
        # Adjust n_iter. 7 was found a good compromise for PCA. See #5299
        n_iter = 7 if n_components < .1 * min(matrix.shape) else 4

    range_matrix, random_matrix, random_proj = randomized_range_finder(matrix, n_random, n_iter,
                                                                       power_iteration_normalizer, random_state, True)
    if one_pass:
        approx_matrix = np.linalg.lstsq(random_matrix.T.dot(range_matrix), random_proj.T.dot(range_matrix), None)[0].T
    else:
        approx_matrix = (matrix.dot(range_matrix)).T.dot(range_matrix)

    eigenvalues, eigenvectors = np.linalg.eig(approx_matrix)

    del approx_matrix
    # eigenvalues indices in decreasing order
    values_order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[values_order]
    eigenvectors = np.dot(range_matrix, eigenvectors)[:, values_order]

    if which == 'SM':
        eigenvalues = lambda_max - eigenvalues

    return eigenvalues[:n_components], eigenvectors[:, :n_components]
