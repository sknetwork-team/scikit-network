"""Sparse-array inputs use the existing CSR-matrix graph backend."""
import unittest

import numpy as np
from scipy import sparse

from sknetwork.classification import DiffusionClassifier
from sknetwork.clustering import Louvain
from sknetwork.data import karate_club, movie_actor
from sknetwork.embedding import LouvainEmbedding, Spectral
from sknetwork.hierarchy import Paris
from sknetwork.linalg.normalizer import get_norms, normalize
from sknetwork.ranking import PageRank
from sknetwork.utils.check import check_adjacency_vector, check_format, has_nonnegative_entries


@unittest.skipUnless(hasattr(sparse, 'csr_array'), 'SciPy sparse arrays require SciPy 1.8 or later.')
class TestSparseArrayInputs(unittest.TestCase):
    def test_formats_and_dtypes(self):
        """Normalize available sparse-array formats without changing values or dtype."""
        for name in ('bsr_array', 'coo_array', 'csc_array', 'csr_array', 'dia_array', 'dok_array', 'lil_array'):
            for dtype in (np.bool_, np.int32, np.float32, np.float64):
                with self.subTest(format=name, dtype=dtype):
                    array = getattr(sparse, name)(np.array([[0, 2, 0], [3, 0, 4]], dtype=dtype))
                    before = array.toarray().copy()
                    actual = check_format(array)
                    self.assertIs(type(actual), sparse.csr_matrix)
                    self.assertEqual(actual.shape, (2, 3))
                    self.assertEqual(actual.dtype, array.dtype)
                    np.testing.assert_array_equal(actual.toarray(), before)
                    np.testing.assert_array_equal(array.toarray(), before)

    def test_empty_sparse_array(self):
        """Keep the existing empty-matrix validation and explicit allowance."""
        array = sparse.csr_array((2, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'empty'):
            check_format(array)
        actual = check_format(array, allow_empty=True)
        self.assertIs(type(actual), sparse.csr_matrix)
        self.assertEqual(actual.shape, array.shape)
        self.assertEqual(actual.nnz, 0)

    def test_wide_indices(self):
        """Small graphs with int64 sparse-array indices use the native matrix index dtype."""
        matrix = karate_club()
        array = sparse.csr_array((matrix.data.copy(), matrix.indices.astype(np.int64),
                                  matrix.indptr.astype(np.int64)), shape=matrix.shape)
        before_indices, before_indptr = array.indices.copy(), array.indptr.copy()
        actual = check_format(array)
        self.assertEqual(actual.indices.dtype, matrix.indices.dtype)
        self.assertEqual(actual.indptr.dtype, matrix.indptr.dtype)
        np.testing.assert_array_equal(actual.toarray(), matrix.toarray())
        np.testing.assert_array_equal(array.indices, before_indices)
        np.testing.assert_array_equal(array.indptr, before_indptr)
        np.testing.assert_array_equal(Louvain().fit_predict(array), Louvain().fit_predict(matrix))

    def test_algorithm_outputs(self):
        """Graph embedding, clustering, ranking and hierarchy match CSR-matrix inputs."""
        matrix = karate_club()
        algorithms = ((lambda: Spectral(2, random_state=0), 'embedding_'),
                      (lambda: LouvainEmbedding(random_state=0), 'embedding_'),
                      (lambda: Louvain(random_state=0), 'probs_'), (PageRank, 'scores_'), (Paris, 'dendrogram_'))
        for make_algorithm, attribute in algorithms:
            expected = getattr(make_algorithm().fit(matrix), attribute)
            for name in ('bsr_array', 'coo_array', 'csc_array', 'csr_array', 'dia_array', 'dok_array', 'lil_array'):
                with self.subTest(algorithm=type(make_algorithm()).__name__, format=name):
                    array = getattr(sparse, name)(matrix)
                    result = make_algorithm().fit(array)
                    actual = getattr(result, attribute)
                    self.assertEqual(actual.dtype, expected.dtype)
                    if sparse.issparse(expected):
                        self.assertIs(type(actual), type(expected))
                        np.testing.assert_allclose(actual.toarray(), expected.toarray())
                    else:
                        np.testing.assert_allclose(actual, expected)
                    np.testing.assert_array_equal(array.toarray(), matrix.toarray())

    def test_sparse_array_classification(self):
        """Semi-supervised labels match the same graph supplied as a sparse matrix."""
        matrix = karate_club()
        labels = np.full(matrix.shape[0], -1)
        labels[0], labels[-1] = 0, 1
        expected = DiffusionClassifier().fit_predict(matrix, labels)
        array = sparse.csr_array(matrix)
        np.testing.assert_array_equal(DiffusionClassifier().fit_predict(array, labels), expected)

    def test_bipartite_embedding(self):
        """A rectangular sparse array preserves the row/column split of a bipartite embedding."""
        matrix = movie_actor()
        expected = Spectral(2, random_state=0).fit(matrix)
        actual = Spectral(2, random_state=0).fit(sparse.coo_array(matrix))
        np.testing.assert_allclose(actual.embedding_row_, expected.embedding_row_)
        np.testing.assert_allclose(actual.embedding_col_, expected.embedding_col_)

    def test_sparse_array_row_norms(self):
        """Sparse-array row reductions match dense arithmetic without mutating signed inputs."""
        dense = np.array([[0, -1000, 0], [1000, 0, -1000], [0, 0, 0]], dtype=np.int16)
        for name in ('bsr_array', 'coo_array', 'csc_array', 'csr_array', 'dia_array', 'dok_array', 'lil_array'):
            for p in (1, 2):
                with self.subTest(format=name, norm=p):
                    array = getattr(sparse, name)(dense)
                    expected = np.linalg.norm(dense.astype(float), ord=p, axis=1)
                    np.testing.assert_allclose(get_norms(array, p=p), expected)
                    normalized = normalize(array, p=p)
                    expected_matrix = dense / np.where(expected == 0, 1, expected)[:, None]
                    np.testing.assert_allclose(normalized.toarray(), expected_matrix)
                    np.testing.assert_array_equal(array.toarray(), dense)
                    self.assertFalse(has_nonnegative_entries(array))
                    self.assertTrue(has_nonnegative_entries(getattr(sparse, name)(np.abs(dense))))

    def test_sparse_array_adjacency_vector(self):
        """One-dimensional sparse vectors have the same row shape as dense prediction vectors."""
        vector = np.array([0., 1., 0.])
        array = sparse.csr_array(vector)
        actual = check_adjacency_vector(array, n=3)
        np.testing.assert_array_equal(actual.toarray(), vector[None, :])
        self.assertEqual(actual.shape, (1, 3))

    def test_legacy_input_contracts(self):
        """Dense and legacy sparse inputs keep their CSR outputs and invalid objects stay rejected."""
        dense = np.eye(3)
        for value in (dense, sparse.csr_matrix(dense), sparse.csc_matrix(dense),
                      sparse.coo_matrix(dense), sparse.lil_matrix(dense)):
            actual = check_format(value)
            self.assertIs(type(actual), sparse.csr_matrix)
            np.testing.assert_array_equal(actual.toarray(), dense)
        vector = np.array([0., 1., 0.])
        self.assertEqual(check_adjacency_vector(vector).shape, (1, 3))
        for value in ([[0, 1], [1, 0]], None, 1):
            with self.assertRaises(TypeError):
                check_format(value)
