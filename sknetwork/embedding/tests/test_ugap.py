"""Correctness tests for UGAP, using only small, locally constructed graphs."""
import multiprocessing
import unittest
from itertools import product
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from sknetwork.embedding import UGAP
from sknetwork.embedding.sgd import sgd
from sknetwork.embedding.ugap import (
    _WORKER_STATE,
    _fuzzy_batch,
    _get_fuzzy_graph,
    _init_worker,
    _piteration_scores,
    _topk_memberships,
)
from sknetwork.linalg import normalize
from sknetwork.linalg.ppr_solver import RandomSurferOperator, get_pagerank
from sknetwork.ranking import PageRank


def ring(n):
    rows = np.arange(n)
    graph = sparse.coo_matrix((np.ones(n), (rows, (rows + 1) % n)), shape=(n, n)).tocsr()
    return graph + graph.T


def optimize(points, row, col, intervals, epochs=4, negatives=0, gamma=0., seed=42, lr=.01):
    points = np.array(points, dtype=np.float64)
    intervals = np.array(intervals, dtype=np.float64)
    next_sample = intervals.copy()
    result = sgd(points.shape[1], epochs, len(points), np.array(row, dtype=np.int32),
                 np.array(col, dtype=np.int32), points, 1., 1., lr, negatives,
                 intervals, next_sample, seed, gamma)
    return np.asarray(result).copy(), next_sample


class TestUGAP(unittest.TestCase):

    def test_fit(self):
        directed = ring(8).astype(float)
        directed[0, 1] = 3.
        directed[2, 3] = 2.
        graphs = [ring(8), directed, directed.toarray(),
                  sparse.block_diag((ring(4), ring(4), sparse.csr_matrix((1, 1))), format='csr'),
                  sparse.eye(5, format='csr')]
        # Defaults exercise n_jobs=-1; reuse the estimator to check repeated fits.
        model = UGAP(n_epochs=5, n_neighbors=15)
        for graph in graphs + [ring(5), ring(20)]:
            with self.subTest(shape=graph.shape, dense=isinstance(graph, np.ndarray)):
                self.assertIs(model.fit(graph), model)
                self.assertIs(model.transform(), model.embedding_)
                result = model.fit_transform(graph)
                self.assertEqual(result.shape, (graph.shape[0], 2))
                self.assertTrue(np.isfinite(result).all())
                self.assertEqual(model.n_neighbors, 15)

    def test_small_and_empty_graphs(self):
        for n in [1, 5]:
            with self.subTest(n=n):
                model = UGAP()
                self.assertIs(model.fit(sparse.csr_matrix((n, n))), model)
                assert_array_equal(model.embedding_, np.zeros((n, 2)))

        with self.assertRaises(ValueError):
            UGAP().fit(sparse.csr_matrix((0, 0)))

        for n, dimensions in [(2, 2), (3, 3), (6, 2), (6, 8)]:
            with self.subTest(n=n, dimensions=dimensions):
                result = UGAP(n_components=dimensions, n_epochs=3, n_jobs=1).fit_transform(ring(n))
                self.assertEqual(result.shape, (n, dimensions))
                self.assertTrue(np.isfinite(result).all())

    def test_invalid_parameters(self):
        counts = ['n_components', 'n_neighbors', 'ppr_n_iter', 'n_epochs', 'negative_sampling_rate']
        positive = ['min_dist', 'spread', 'lr', 'ppr_tol']
        cases = list(product(counts, [0, -1, True, 1.5, np.nan]))
        cases += list(product(positive, [0, -1, True, np.nan, np.inf]))
        cases += [('gamma', -1), ('gamma', np.nan), ('gamma', np.inf),
                  ('damping_factor', 0), ('damping_factor', 1), ('damping_factor', np.nan),
                  ('n_jobs', 0), ('n_jobs', -2), ('n_jobs', True), ('n_jobs', 1.5)]
        for name, value in cases:
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    UGAP(**{name: value})

    def test_reproducibility(self):
        for n, dimensions in [(2, 2), (3, 3), (8, 2), (8, 10)]:
            with self.subTest(n=n, dimensions=dimensions):
                graph = ring(n)
                first = UGAP(n_components=dimensions, n_epochs=5, n_jobs=1, random_state=17).fit_transform(graph)
                second = UGAP(n_components=dimensions, n_epochs=5, n_jobs=1, random_state=17).fit_transform(graph)
                assert_allclose(first, second, atol=1e-10)

        points = np.arange(16, dtype=float).reshape(8, 2)
        first, _ = optimize(points, [0, 2], [1, 3], [1, 1], negatives=5, gamma=1, seed=10)
        same, _ = optimize(points, [0, 2], [1, 3], [1, 1], negatives=5, gamma=1, seed=10)
        different, _ = optimize(points, [0, 2], [1, 3], [1, 1], negatives=5, gamma=1, seed=20)
        assert_array_equal(first, same)
        self.assertFalse(np.allclose(first, different))

        # Seeding initialization must not reset the caller's process-wide generator.
        before = np.random.get_state()
        UGAP(n_jobs=1, n_epochs=2).fit(ring(8))
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_ppr_matches_existing_solver(self):
        for undirected, damping, (n_iter, tol) in product(
                [False, True], [.3, .85], [(1, 1e-12), (20, 1e-12), (100, .1)]):
            with self.subTest(undirected=undirected, damping=damping, n_iter=n_iter, tol=tol):
                # Weighted directed component, dangling node, and isolated node.
                graph = sparse.csr_matrix([[0, 2, 1, 0, 0], [1, 0, 3, 0, 0],
                                           [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0.]])
                if undirected:
                    graph = graph + graph.T
                restart = 1 - damping * np.asarray(graph.sum(axis=1)).ravel().astype(bool)
                for source in range(5):
                    seeds = np.eye(5)[source]
                    operator = RandomSurferOperator(graph, seeds, damping)
                    actual = _piteration_scores(operator.a, restart, source, n_iter, tol)
                    expected = get_pagerank(graph, seeds, damping, n_iter, tol, 'piteration')
                    assert_allclose(actual, expected, atol=1e-14)
                    assert_allclose(actual.sum(), 1)

    def test_neighbors(self):
        for k in [1, 2, 10]:
            with self.subTest(k=k):
                scores = np.array([.5, .3, .15, .05, 0])
                before = scores.copy()
                indices, weights = _topk_memberships(scores, 0, k)
                count = min(k, 3)
                self.assertEqual(set(indices), set(range(1, count + 1)))
                distances = -np.log(scores[indices] + 1e-9)
                assert_allclose(weights, np.exp(-distances / (distances.mean() + 1e-9)))
                assert_array_equal(scores, before)

        self.assertEqual(_topk_memberships(np.array([1.0, 0, 0]), 0, 2), ([], []))
        indices, weights = _topk_memberships(np.array([.4, .2, .2, .2]), 0, 2)
        self.assertEqual(len(set(indices)), 2)
        self.assertNotIn(0, indices)
        assert_allclose(weights[0], weights[1])

    def test_fuzzy_graph(self):
        graph = ring(6)
        pagerank = PageRank(n_iter=10, tol=1e-12)
        directed = np.zeros((6, 6))
        for source in range(6):
            scores = pagerank.fit_predict(graph, weights={source: 1})
            scores[source] = 0
            selected = np.argsort(-scores)[:5]
            distances = -np.log(scores[selected] + 1e-9)
            directed[source, selected] = np.exp(-distances / (distances.mean() + 1e-9))
        expected = directed + directed.T - directed * directed.T
        expected /= expected.sum(axis=1, keepdims=True)
        actual = _get_fuzzy_graph(graph, pagerank, 5, n_jobs=1)
        assert_allclose(actual.toarray(), expected)
        assert_array_equal(actual.diagonal(), np.zeros(6))

    def test_parallel(self):
        for n in [499, 500]:
            with self.subTest(n=n):
                # Use real spawned processes even on systems whose default is fork.
                graph = ring(n)
                pagerank = PageRank(n_iter=3)
                expected = _get_fuzzy_graph(graph, pagerank, 4, n_jobs=1)
                with patch('sknetwork.embedding.ugap.Pool', wraps=multiprocessing.get_context('spawn').Pool) as pool:
                    actual = _get_fuzzy_graph(graph, pagerank, 4, n_jobs=2)
                self.assertEqual(pool.call_count, int(n >= 500))
                assert_allclose(actual.toarray(), expected.toarray(), atol=1e-14)

        graph = ring(500)
        with patch('sknetwork.embedding.ugap.os.cpu_count', return_value=2), \
                patch('sknetwork.embedding.ugap.Pool', wraps=multiprocessing.get_context('spawn').Pool) as pool:
            model = UGAP(n_jobs=-1, n_epochs=2)
            actual = _get_fuzzy_graph(graph, PageRank(n_iter=3), 4, model.n_jobs)
        self.assertEqual(pool.call_args.args[0], 2)
        assert_allclose(actual.toarray(), _get_fuzzy_graph(graph, PageRank(n_iter=3), 4, 1).toarray())

    def test_init_worker_in_process(self):
        graph = ring(6)
        pagerank = PageRank(n_iter=3)
        n = graph.shape[0]
        transition_t = (pagerank.damping_factor * normalize(graph)).T.tocsr()
        restart_scale = 1. - pagerank.damping_factor * graph.dot(np.ones(n)).astype(bool)
        self.addCleanup(_WORKER_STATE.clear)
        _init_worker(transition_t.indptr, transition_t.indices, transition_t.data,
                     n, restart_scale, pagerank.n_iter, pagerank.tol, 4)
        assert_allclose(_WORKER_STATE['transition_t'].toarray(), transition_t.toarray())
        assert_allclose(_WORKER_STATE['restart_scale'], np.asarray(restart_scale))
        self.assertEqual(_WORKER_STATE['n_iter'], pagerank.n_iter)
        self.assertEqual(_WORKER_STATE['tol'], pagerank.tol)
        self.assertEqual(_WORKER_STATE['n_neighbors'], 4)

    def test_fuzzy_batch_in_process(self):
        graph = ring(6)
        pagerank = PageRank(n_iter=3)
        n = graph.shape[0]
        transition_t = (pagerank.damping_factor * normalize(graph)).T.tocsr()
        restart_scale = 1. - pagerank.damping_factor * graph.dot(np.ones(n)).astype(bool)
        self.addCleanup(_WORKER_STATE.clear)
        _init_worker(transition_t.indptr, transition_t.indices, transition_t.data,
                     n, restart_scale, pagerank.n_iter, pagerank.tol, 4)
        sources = [0, 1, 5]
        rows, cols, vals = _fuzzy_batch(sources)
        expected_rows, expected_cols, expected_vals = [], [], []
        for source in sources:
            scores = _piteration_scores(transition_t, restart_scale, source,
                                        pagerank.n_iter, pagerank.tol)
            indices, memberships = _topk_memberships(scores, source, 4)
            expected_rows.extend([source] * len(indices))
            expected_cols.extend(indices)
            expected_vals.extend(memberships)
        self.assertEqual(rows, expected_rows)
        self.assertEqual(cols, expected_cols)
        assert_allclose(vals, expected_vals)
        self.assertNotIn(0, [c for r, c in zip(rows, cols) if r == 0])

        self.assertEqual(_fuzzy_batch([]), ([], [], []))

        # Isolated node contributes no entries.
        isolated = sparse.block_diag((ring(4), sparse.csr_matrix((1, 1))), format='csr')
        n_iso = isolated.shape[0]
        transition_iso = (pagerank.damping_factor * normalize(isolated)).T.tocsr()
        restart_iso = 1. - pagerank.damping_factor * isolated.dot(np.ones(n_iso)).astype(bool)
        _init_worker(transition_iso.indptr, transition_iso.indices, transition_iso.data,
                     n_iso, restart_iso, pagerank.n_iter, pagerank.tol, 4)
        rows, cols, vals = _fuzzy_batch([4])
        # An isolated source may still reach others via restart; just check consistency
        scores = _piteration_scores(transition_iso, restart_iso, 4,
                                    pagerank.n_iter, pagerank.tol)
        indices, memberships = _topk_memberships(scores, 4, 4)
        self.assertEqual(rows, [4] * len(indices))
        self.assertEqual(cols, indices)
        assert_allclose(vals, memberships)
        self.assertNotIn(4, cols)

    def test_get_fuzzy_graph_parallel_in_process(self):
        class DummyPool:
            """Run Pool.map inline so coverage sees the parallel branch."""
            def __init__(self, n_workers, initializer=None, initargs=()):
                self.n_workers = n_workers
                self.initializer = initializer
                self.initargs = initargs

            def __enter__(self):
                if self.initializer is not None:
                    self.initializer(*self.initargs)
                return self

            def __exit__(self, *args):
                _WORKER_STATE.clear()
                return False

            def map(self, fn, chunks):
                return [fn(chunk) for chunk in chunks]

        self.addCleanup(_WORKER_STATE.clear)
        graph = ring(8)
        pagerank = PageRank(n_iter=3)
        expected = _get_fuzzy_graph(graph, pagerank, 4, n_jobs=1)
        with patch('sknetwork.embedding.ugap.PARALLEL_MIN_N', 5), \
                patch('sknetwork.embedding.ugap.Pool', DummyPool):
            actual = _get_fuzzy_graph(graph, pagerank, 4, n_jobs=2)
        assert_allclose(actual.toarray(), expected.toarray(), atol=1e-14)

        # n_jobs=-1 resolves worker count from cpu_count.
        with patch('sknetwork.embedding.ugap.PARALLEL_MIN_N', 5), \
                patch('sknetwork.embedding.ugap.os.cpu_count', return_value=2), \
                patch('sknetwork.embedding.ugap.Pool', DummyPool) as pool_cls:
            seen = {}
            orig_init = pool_cls.__init__

            def capture_init(pool_self, n_workers, initializer=None, initargs=()):
                seen['n_workers'] = n_workers
                orig_init(pool_self, n_workers, initializer, initargs)

            with patch.object(pool_cls, '__init__', capture_init):
                actual = _get_fuzzy_graph(graph, pagerank, 4, n_jobs=-1)
        self.assertEqual(seen.get('n_workers'), 2)
        assert_allclose(actual.toarray(), expected.toarray(), atol=1e-14)

    def test_curve_fit_failure(self):
        with patch('sknetwork.embedding.ugap.curve_fit',
                   side_effect=RuntimeError('no fit')):
            with self.assertRaisesRegex(ValueError, 'Failed to fit'):
                UGAP(n_epochs=2, n_jobs=1).fit(ring(8))

    def test_sgd_attraction(self):
        points = np.array([[0., 0], [2., 1]])
        actual, _ = optimize(points, [0], [1], [1], epochs=2)
        self.assertLess(np.linalg.norm(actual[0] - actual[1]), np.linalg.norm(points[0] - points[1]))
        assert_allclose(actual.mean(axis=0), points.mean(axis=0))

    def test_sgd_repulsion(self):
        # A self-edge has zero attraction. All non-self negative draws repel node 0.
        points = np.array([[0., 0], [1., 0], [2., 0], [3., 0]])
        actual, _ = optimize(points, [0], [0], [1], epochs=2, negatives=100, gamma=1.)
        self.assertLess(actual[0, 0], 0)
        assert_array_equal(actual[1:], points[1:])
        assert_array_equal(actual[:, 1], points[:, 1])

        points = [[0., 0], [1., 2], [3., 4]]
        without, _ = optimize(points, [0], [1], [1])
        with_samples, _ = optimize(points, [0], [1], [1], negatives=20, gamma=0)
        assert_array_equal(without, with_samples)

    def test_sgd_scheduler(self):
        points = np.array([[0., 0], [1., 2], [3., 1], [-1., 2]])
        row, col = [0, 1, 2, 0], [1, 2, 3, 3]
        intervals = np.array([1., 2.5, .5, 20.])
        epochs = 9
        # Build a schedule independently, then apply vectorized pair updates.
        schedule = []
        counts = np.zeros(4, dtype=int)
        for epoch in range(epochs):
            for edge, interval in enumerate(intervals):
                if epoch >= (counts[edge] + 1) * interval:
                    schedule.append((epoch, edge))
                    counts[edge] += 1
        expected = points.copy()
        lr = float(np.float32(.01))  # public Cython signature uses a C float
        for epoch, edge in schedule:
            i, j = row[edge], col[edge]
            delta = expected[i] - expected[j]
            step = lr * (1 - epoch / epochs) * (-2 * delta / (1 + delta @ delta))
            expected[i] += step
            expected[j] -= step
        actual, next_sample = optimize(points, row, col, intervals, epochs=epochs)
        assert_allclose(actual, expected, atol=1e-12)
        assert_allclose(next_sample, (counts + 1) * intervals)
        self.assertEqual(counts[-1], 0)

    def test_sgd_edge_cases(self):
        for separation in [0.0, 1e-12]:
            with self.subTest(separation=separation):
                actual, _ = optimize([[0, 0], [separation, 0]], [0], [1], [1], negatives=10, gamma=1)
                self.assertTrue(np.isfinite(actual).all())

        for shape, n, dimensions in [((3, 1), 3, 2), ((2, 2), 3, 2)]:
            with self.subTest(shape=shape, n=n, dimensions=dimensions):
                with self.assertRaisesRegex(ValueError, 'Embedding shape'):
                    sgd(dimensions, 2, n, np.array([0], dtype=np.int32), np.array([1], dtype=np.int32),
                        np.zeros(shape), 1., 1., .01, 1, np.ones(1), np.ones(1), 42, 1.)
