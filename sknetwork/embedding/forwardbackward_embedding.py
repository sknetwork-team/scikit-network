#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 17:16:22 2018

@author: Nathan de Lara <ndelara@enst.fr>

Part of this code was adapted from the scikit-learn project.
"""

import numpy as np

from scipy import sparse, errstate, sqrt, isinf, linalg


class ForwardBackwardEmbedding:
    """Forward and Backward embeddings for non-linear dimensionality reduction.

    Parameters
    -----------
    n_components: int, optional
        The dimension of the projected subspace (default=2).

    Attributes
    ----------
    embedding_ : array, shape = (n_samples, n_components)
        Forward embedding of the training matrix.

    backward_embedding_ : array, shape = (n_samples, n_components)
        Backward embedding of the training matrix.

    singular_values_ : array, shape = (n_components)
        Singular values of the training matrix

    References
    ----------
    - Bonald, De Lara. "The Forward-Backward Embedding of Directed Graphs."
    """

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.embedding_ = None
        self.backward_embedding_ = None
        self.singular_values_ = None

    def fit(self, adjacency_matrix, tol=1e-6, n_iter='auto',
            power_iteration_normalizer='auto', random_state=None):
        """Fits the model from data in adjacency_matrix.

        Parameters
        ----------
        adjacency_matrix: array-like, shape = (n, m)
            Adjacency matrix, where n = m = |V| for a standard graph,
            n = |V1|, m = |V2| for a bipartite graph.

        tol: float, optional
            Tolerance for pseudo-inverse of singular values (default=1e-6).

        n_iter: int or 'auto' (default is 'auto')
            Number of power iterations. It can be used to deal with very noisy
            problems. When 'auto', it is set to 4, unless `n_components` is small
            (< .1 * min(X.shape)) `n_iter` in which case is set to 7.
            This improves precision with few components.

        power_iteration_normalizer: 'auto' (default), 'QR', 'LU', 'none'
            Whether the power iterations are normalized with step-by-step
            QR factorization (the slowest but most accurate), 'none'
            (the fastest but numerically unstable when `n_iter` is large, e.g.
            typically 5 or larger), or 'LU' factorization (numerically stable
            but can lose slightly in accuracy). The 'auto' mode applies no
            normalization if `n_iter`<=2 and switches to LU otherwise.

        random_state: int, RandomState instance or None, optional (default=None)
            The seed of the pseudo random number generator to use when shuffling
            the data.  If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random number
            generator; If None, the random number generator is the RandomState
            instance used by `np.random`.

        Returns
        -------
        self

        """
        if type(adjacency_matrix) == sparse.csr_matrix:
            adj_matrix = adjacency_matrix
        elif type(adjacency_matrix) == np.ndarray:
            adj_matrix = sparse.csr_matrix(adjacency_matrix)
        else:
            raise TypeError(
                "The argument should be a NumPy array or a SciPy Compressed Sparse Row matrix.")
        n_nodes, m_nodes = adj_matrix.shape

        # out-degree vector
        dou = adj_matrix.sum(axis=1).flatten()
        # in-degree vector
        din = adj_matrix.sum(axis=0).flatten()

        with errstate(divide='ignore'):
            dou_sqrt = 1.0 / sqrt(dou)
            din_sqrt = 1.0 / sqrt(din)
        dou_sqrt[isinf(dou_sqrt)] = 0
        din_sqrt[isinf(din_sqrt)] = 0
        # pseudo inverse square-root out-degree matrix
        dhou = sparse.spdiags(dou_sqrt, [0], n_nodes, n_nodes, format='csr')
        # pseudo inverse square-root in-degree matrix
        dhin = sparse.spdiags(din_sqrt, [0], m_nodes, m_nodes, format='csr')

        laplacian = dhou.dot(adj_matrix.dot(dhin))
        u, sigma, vt = randomized_svd(laplacian, self.n_components, n_iter=n_iter,
                                      power_iteration_normalizer=power_iteration_normalizer, random_state=random_state)

        self.singular_values_ = sigma

        gamma = 1 - sigma ** 2
        gamma_sqrt = np.diag(np.piecewise(gamma, [gamma > tol, gamma <= tol], [lambda x: 1 / np.sqrt(x), 0]))
        self.embedding_ = dhou.dot(u).dot(gamma_sqrt)
        self.backward_embedding_ = dhin.dot(vt.T).dot(gamma_sqrt)

        return self


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
    if sparse.issparse(a) or sparse.issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)


def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.
    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    Parameters
    ----------
    u, v : ndarray
        u and v are the output of `linalg.svd` or
        `sklearn.utils.extmath.randomized_svd`, with matching inner dimensions
        so one can compute `np.dot(u * s, v)`.
    u_based_decision : boolean, (default=True)
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.
    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.
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


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, np.integer):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def randomized_range_finder(A, size, n_iter,
                            power_iteration_normalizer='auto',
                            random_state=None):
    """Computes an orthonormal matrix whose range approximates the range of A.
    Parameters
    ----------
    A : 2D array
        The input data matrix
    size : integer
        Size of the return array
    n_iter : integer
        Number of power iterations used to stabilize the result
    power_iteration_normalizer : 'auto' (default), 'QR', 'LU', 'none'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter`<=2 and switches to LU otherwise.
        .. versionadded:: 0.18
    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.
    Returns
    -------
    Q : 2D array
        A (size x size) projection matrix, the range of which
        approximates well the range of the input matrix A.
    Notes
    -----
    Follows Algorithm 4.3 of
    Finding structure with randomness: Stochastic algorithms for constructing
    approximate matrix decompositions
    Halko, et al., 2009 (arXiv:909) http://arxiv.org/pdf/0909.4061
    An implementation of a randomized algorithm for principal component
    analysis
    A. Szlam et al. 2014
    """
    random_state = check_random_state(random_state)

    # Generating normal random vectors with shape: (A.shape[1], size)
    Q = random_state.normal(size=(A.shape[1], size))
    if A.dtype.kind == 'f':
        # Ensure f32 is preserved as f32
        Q = Q.astype(A.dtype, copy=False)

    # Deal with "auto" mode
    if power_iteration_normalizer == 'auto':
        if n_iter <= 2:
            power_iteration_normalizer = 'none'
        else:
            power_iteration_normalizer = 'LU'

    # Perform power iterations with Q to further 'imprint' the top
    # singular vectors of A in Q
    for i in range(n_iter):
        if power_iteration_normalizer == 'none':
            Q = safe_sparse_dot(A, Q)
            Q = safe_sparse_dot(A.T, Q)
        elif power_iteration_normalizer == 'LU':
            Q, _ = linalg.lu(safe_sparse_dot(A, Q), permute_l=True)
            Q, _ = linalg.lu(safe_sparse_dot(A.T, Q), permute_l=True)
        elif power_iteration_normalizer == 'QR':
            Q, _ = linalg.qr(safe_sparse_dot(A, Q), mode='economic')
            Q, _ = linalg.qr(safe_sparse_dot(A.T, Q), mode='economic')

    # Sample the range of A using by linear projection of Q
    # Extract an orthonormal basis
    Q, _ = linalg.qr(safe_sparse_dot(A, Q), mode='economic')
    return Q


def randomized_svd(M, n_components, n_oversamples=10, n_iter='auto', transpose='auto',
                   power_iteration_normalizer='auto', flip_sign=True, random_state=0):
    """Computes a truncated randomized SVD
        Parameters
        ----------
        M : ndarray or sparse matrix
            Matrix to decompose
        n_components : int
            Number of singular values and vectors to extract.
        n_oversamples : int (default is 10)
            Additional number of random vectors to sample the range of M so as
            to ensure proper conditioning. The total number of random vectors
            used to find the range of M is n_components + n_oversamples. Smaller
            number can improve speed but can negatively impact the quality of
            approximation of singular vectors and singular values.
        n_iter : int or 'auto' (default is 'auto')
            Number of power iterations. It can be used to deal with very noisy
            problems. When 'auto', it is set to 4, unless `n_components` is small
            (< .1 * min(X.shape)) `n_iter` in which case is set to 7.
            This improves precision with few components.
            .. versionchanged:: 0.18
        power_iteration_normalizer : 'auto' (default), 'QR', 'LU', 'none'
            Whether the power iterations are normalized with step-by-step
            QR factorization (the slowest but most accurate), 'none'
            (the fastest but numerically unstable when `n_iter` is large, e.g.
            typically 5 or larger), or 'LU' factorization (numerically stable
            but can lose slightly in accuracy). The 'auto' mode applies no
            normalization if `n_iter`<=2 and switches to LU otherwise.
            .. versionadded:: 0.18
        transpose : True, False or 'auto' (default)
            Whether the algorithm should be applied to M.T instead of M. The
            result should approximately be the same. The 'auto' mode will
            trigger the transposition if M.shape[1] > M.shape[0] since this
            implementation of randomized SVD tend to be a little faster in that
            case.
            .. versionchanged:: 0.18
        flip_sign : boolean, (True by default)
            The output of a singular value decomposition is only unique up to a
            permutation of the signs of the singular vectors. If `flip_sign` is
            set to `True`, the sign ambiguity is resolved by making the largest
            loadings for each component in the left singular vectors positive.
        random_state : int, RandomState instance or None, optional (default=None)
            The seed of the pseudo random number generator to use when shuffling
            the data.  If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random number
            generator; If None, the random number generator is the RandomState
            instance used by `np.random`.
        Notes
        -----
        This algorithm finds a (usually very good) approximate truncated
        singular value decomposition using randomization to speed up the
        computations. It is particularly fast on large matrices on which
        you wish to extract only a small number of components. In order to
        obtain further speed up, `n_iter` can be set <=2 (at the cost of
        loss of precision).
        References
        ----------
        * Finding structure with randomness: Stochastic algorithms for constructing
          approximate matrix decompositions
          Halko, et al., 2009 http://arxiv.org/abs/arXiv:0909.4061
        * A randomized algorithm for the decomposition of matrices
          Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert
        * An implementation of a randomized algorithm for principal component
          analysis
          A. Szlam et al. 2014
        """

    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples
    n_samples, n_features = M.shape

    if n_iter == 'auto':
        # Checks if the number of iterations is explicitly specified
        # Adjust n_iter. 7 was found a good compromise for PCA. See #5299
        n_iter = 7 if n_components < .1 * min(M.shape) else 4

    if transpose == 'auto':
        transpose = n_samples < n_features
    if transpose:
        # this implementation is a bit faster with smaller shape[1]
        M = M.T

    Q = randomized_range_finder(M, n_random, n_iter,
                                power_iteration_normalizer, random_state)

    # project M to the (k + p) dimensional space using the basis vectors
    B = safe_sparse_dot(Q.T, M)

    # compute the SVD on the thin matrix: (k + p) wide
    Uhat, s, V = linalg.svd(B, full_matrices=False)

    del B
    U = np.dot(Q, Uhat)

    if flip_sign:
        if not transpose:
            U, V = svd_flip(U, V)
        else:
            # In case of transpose u_based_decision=false
            # to actually flip based on u and not v.
            U, V = svd_flip(U, V, u_based_decision=False)

    if transpose:
        # transpose back the results according to the input convention
        return V[:n_components, :].T, s[:n_components], U[:, :n_components].T
    else:
        return U[:, :n_components], s[:n_components], V[:n_components, :]


