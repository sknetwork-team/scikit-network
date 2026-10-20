import os
from typing import Union
from multiprocessing import Pool

import numpy as np
from scipy import sparse
from scipy.optimize import curve_fit

from sknetwork.embedding.base import BaseEmbedding
from sknetwork.linalg import normalize
from sknetwork.utils.format import get_adjacency
from sknetwork.ranking import PageRank
from sknetwork.embedding.spectral import Spectral
from sknetwork.embedding.sgd import sgd

# Minimum number of nodes for parallelism gains
PARALLEL_MIN_N = 500


def _piteration_scores(transition_t: sparse.csr_matrix, restart_scale: np.ndarray, source: int,
                       n_iter: int, tol: float) -> np.ndarray:
    """Single-source power iteration with a precomputed transposed transition.

    Replicates the ``'piteration'`` branch of ``get_pagerank`` while reusing the
    operator built once per graph instead of once per source. ``restart_scale``
    holds ``1`` on dangling nodes and ``1 - damping_factor`` elsewhere, so that
    ``b = restart_scale * seeds`` matches ``RandomSurferOperator.b``.
    """
    n = transition_t.shape[0]
    seeds = np.zeros(n)
    seeds[source] = 1.
    base = restart_scale * seeds
    scores = base.copy()
    for _ in range(n_iter):
        updated = transition_t.dot(scores) + base * scores.sum()
        updated /= updated.sum()
        if np.linalg.norm(scores - updated, ord=1) < tol:
            break
        scores = updated
    return scores / scores.sum()


def _topk_memberships(scores: np.ndarray, source: int, n_neighbors: int):
    """Top-k neighbor indices and fuzzy memberships for one source."""
    eps = 1e-9
    scores = scores.copy()
    scores[source] = 0
    positive = np.flatnonzero(scores > 0)
    if len(positive) == 0:
        return [], []
    n_selected = min(n_neighbors, len(positive))
    positive_scores = scores[positive]
    selected = np.argpartition(-positive_scores, n_selected - 1)[:n_selected]
    indices = positive[selected]
    dists = -np.log(positive_scores[selected] + eps)
    sigma = np.mean(dists) + eps
    memberships = np.exp(-dists / sigma)
    return indices.tolist(), memberships.tolist()


_WORKER_STATE = {}


def _init_worker(indptr, indices, data, n, restart_scale,
                 n_iter, tol, n_neighbors):
    """Share the precomputed PPR operator with a source (pool worker)."""
    _WORKER_STATE['transition_t'] = sparse.csr_matrix(
        (np.asarray(data), np.asarray(indices), np.asarray(indptr)), shape=(n, n))
    _WORKER_STATE['restart_scale'] = np.asarray(restart_scale)
    _WORKER_STATE['n_iter'] = n_iter
    _WORKER_STATE['tol'] = tol
    _WORKER_STATE['n_neighbors'] = n_neighbors


def _fuzzy_batch(sources):
    """Top-k neighborhoods for a batch of sources."""
    state = _WORKER_STATE
    rows, cols, vals = [], [], []
    for source in sources:
        scores = _piteration_scores(state['transition_t'], state['restart_scale'],
                                    int(source), state['n_iter'], state['tol'])
        indices, memberships = _topk_memberships(scores, int(source),
                                                 state['n_neighbors'])
        rows.extend([int(source)] * len(indices))
        cols.extend(indices)
        vals.extend(memberships)
    return rows, cols, vals


def _get_fuzzy_graph(adjacency: sparse.csr_matrix, pagerank: PageRank, n_neighbors: int,
                     n_jobs: int = -1) -> sparse.csr_matrix:
    """Build the fuzzy graph from the top-k personalized PageRank scores, via power iteration.
    """
    n = adjacency.shape[0]

    transition_t = (pagerank.damping_factor * normalize(adjacency)).T.tocsr()
    restart_scale = 1. - pagerank.damping_factor * adjacency.dot(np.ones(n)).astype(bool)

    rows: list = []
    cols: list = []
    vals: list = []

    if n_jobs == -1:
        n_workers = os.cpu_count() or 1
    else:
        n_workers = n_jobs

    if n_workers > 1 and n >= PARALLEL_MIN_N:
        chunks = [chunk.tolist() for chunk in np.array_split(np.arange(n), 4 * n_workers)]
        with Pool(n_workers, initializer=_init_worker,
                  initargs=(transition_t.indptr, transition_t.indices, transition_t.data,
                            n, restart_scale, pagerank.n_iter, pagerank.tol,
                            n_neighbors)) as pool:
            for batch_rows, batch_cols, batch_vals in pool.map(_fuzzy_batch, chunks):
                rows.extend(batch_rows)
                cols.extend(batch_cols)
                vals.extend(batch_vals)
    else:
        for source in range(n):
            scores = _piteration_scores(transition_t, restart_scale, source,
                                        pagerank.n_iter, pagerank.tol)
            indices, memberships = _topk_memberships(scores, source, n_neighbors)
            rows.extend([source] * len(indices))
            cols.extend(indices)
            vals.extend(memberships)

    graph = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n))
    prod = graph.multiply(graph.T)
    graph = graph + graph.T - prod
    return normalize(graph.tocsr(), p=1)


class UGAP(BaseEmbedding):
    r"""Low dimensional projection of graph data based on Personalized PageRank similarity.

    Parameters
    ----------
    n_components : int
        Dimension of the embedding space. Seeded random initialization is used
        when the graph has too few nodes for spectral initialization.
    n_neighbors : int
        Maximum number of personalized PageRank neighbors per node.
    min_dist : float
        Minimum distance used to fit the low-dimensional membership curve.
    spread : float
        Scale used to fit the low-dimensional membership curve.
    damping_factor : float
        Probability to continue the random walk in personalized PageRank.
    ppr_n_iter : int
        Number of power iterations per personalized PageRank solve.
    ppr_tol : float
        L1 early-stop tolerance of the power iteration.
    n_epochs : int
        Number of optimization epochs.
    negative_sampling_rate : int
        Number of negative samples per positive edge update.
    gamma : float
        Weighting applied to negative samples (repulsion strength) in the optimization objective.
        A value of 0 disables repulsion entirely.
    random_state : int
        Seed for reproducible initialization and optimization.
    lr : float
        Initial learning rate for stochastic optimization.
    n_jobs : int, optional
        Number of processes for the per-node PageRank solves. ``-1`` uses all CPUs. Small graphs (fewer
        than 500 nodes) always run serially.

    Attributes
    ----------
    embedding_ : np.ndarray, shape = (n_nodes, n_components)
        Embedding of the nodes.

    Example
    -------
    >>> from sknetwork.embedding import UGAP
    >>> from sknetwork.data import karate_club
    >>> ugap = UGAP(n_components=2, random_state=42)
    >>> adjacency = karate_club()
    >>> embedding = ugap.fit_transform(adjacency)
    >>> embedding.shape
    (34, 2)
    """
    def __init__(self, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1, spread: float = 1.0,
                 damping_factor: float = 0.7, ppr_n_iter: int = 3, n_epochs: int = 1000,
                 negative_sampling_rate: int = 5, gamma: float = 1.0, random_state: int = 42, lr: float = 0.8,
                 ppr_tol: float = 1e-2, n_jobs: int = -1):
        super(UGAP, self).__init__()

        integer_parameters = {
            'n_components': n_components,
            'n_neighbors': n_neighbors,
            'ppr_n_iter': ppr_n_iter,
            'n_epochs': n_epochs,
            'negative_sampling_rate': negative_sampling_rate,
        }
        for name, value in integer_parameters.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f'{name} must be a positive integer.')

        positive_parameters = {
            'min_dist': min_dist,
            'spread': spread,
            'lr': lr,
            'ppr_tol': ppr_tol,
        }
        for name, value in positive_parameters.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) \
                    or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a positive finite number.')

        if isinstance(gamma, (bool, np.bool_)) or not isinstance(gamma, (int, float, np.number)) \
                or not np.isfinite(gamma) or gamma < 0:
            raise ValueError('gamma must be a non-negative finite number.')

        if not isinstance(damping_factor, (int, float, np.number)) or not np.isfinite(damping_factor) \
                or not 0 < damping_factor < 1:
            raise ValueError('damping_factor must be between 0 and 1.')


        if isinstance(n_jobs, (bool, np.bool_)) or not isinstance(n_jobs, (int, np.integer)) or (
                n_jobs <= 0 and n_jobs != -1):
            raise ValueError('n_jobs must be -1 or a positive integer.')

        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.damping_factor = damping_factor 
        self.ppr_n_iter = ppr_n_iter
        self.ppr_tol = ppr_tol
        self.n_jobs = n_jobs
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.min_dist = min_dist
        self.spread = spread
        self.embedding_ = None
        self.negative_sampling_rate = negative_sampling_rate
        self.gamma = gamma
        self.epochs_per_sample = None
        self.lr = lr

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray]) -> 'UGAP':

        adjacency, _ = get_adjacency(input_matrix, allow_empty=True)

        n = adjacency.shape[0]

        # deal with edge cases
        if n == 0:
            raise ValueError('The graph must contain at least one node.')
        if n == 1 or adjacency.nnz == 0:
            self.embedding_ = np.zeros((n, self.n_components))
            return self

        # creating pagerank similarity matrix
        pagerank = PageRank(damping_factor=self.damping_factor, solver='piteration',
                            n_iter=self.ppr_n_iter, tol=self.ppr_tol)
        graph = _get_fuzzy_graph(adjacency, pagerank, min(self.n_neighbors, n - 1),
                                 self.n_jobs)

        if graph.nnz == 0:
            self.embedding_ = np.zeros((n, self.n_components))
            return self

        seed = int(self.random_state) % (2 ** 32) if isinstance(
            self.random_state, (int, np.integer)) else 0
        # Spectral needs at least n_components + 2 nodes. A random fallback
        # preserves the requested shape, including for two-node graphs.
        if self.n_components >= n - 1:
            low_dim = np.random.RandomState(seed).normal(scale=1e-4, size=(n, self.n_components))
        else:
            spectral = Spectral(self.n_components, random_state=seed)
            low_dim = np.asarray(spectral.fit_transform(graph), dtype=np.float64)

        xv = np.linspace(0, self.spread * 3, 500)
        yv = np.zeros(xv.shape)
        yv[xv < self.min_dist] = 1.0
        yv[xv >= self.min_dist] = np.exp(-(xv[xv >= self.min_dist] - self.min_dist)/self.spread)

        def curve(x, a, b):
            return 1.0 / (1.0 + a * x ** (2 * b))

        try:
            params, _ = curve_fit(curve, xv, yv)
        except RuntimeError as e:
            raise ValueError(f'Failed to fit the membership curve. Check min_dist ({self.min_dist}) '
                             f'and spread ({self.spread}) parameters.') from e
        a = params[0] 
        b = params[1]

        # edge sampling rate
        graph = graph.tocoo()
        graph.sum_duplicates()
        graph.eliminate_zeros()

        weights = graph.data
        self.epochs_per_sample = np.full(weights.shape[0], -1.0, dtype=np.float64)
        n_samples = self.n_epochs * (weights / weights.max())
        positive = n_samples > 0
        self.epochs_per_sample[positive] = float(self.n_epochs) / np.float64(n_samples[positive])

        # edge scheduler
        self.epoch_of_next_sample = np.copy(self.epochs_per_sample)

        # SGD
        self.embedding_ = np.asarray(sgd(self.n_components, self.n_epochs, n, graph.row, graph.col,
                                         low_dim, a, b, self.lr, self.negative_sampling_rate,
                                         self.epochs_per_sample, self.epoch_of_next_sample,
                                         seed, float(self.gamma)))

        return self
