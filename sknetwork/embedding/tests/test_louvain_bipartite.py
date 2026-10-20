"""Square biadjacency inputs and refit state for Louvain embeddings."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from sknetwork.embedding import LouvainEmbedding
from sknetwork.embedding.louvain_embedding import reindex_labels


def biadjacency():
    return sparse.csr_matrix([[3, 3, 0], [0, 0, 3], [0, 0, 3]])


@pytest.mark.parametrize('reverse_rows', [False, True])
@pytest.mark.parametrize('reverse_columns', [False, True])
def test_louvain_forced_square_bipartite(reverse_rows, reverse_columns):
    rows = [2, 1, 0] if reverse_rows else [0, 1, 2]
    columns = [2, 1, 0] if reverse_columns else [0, 1, 2]
    matrix = biadjacency()[rows][:, columns]
    model = LouvainEmbedding().fit(matrix, force_bipartite=True)
    # Default removal retains the cluster containing two column nodes.
    expected_row = np.array([[1.0], [0.0], [0.0]])[rows]
    expected_column = np.array([[1.0], [1.0], [0.0]])[columns]
    assert_allclose(model.embedding_, expected_row)
    assert model.embedding_row_ is model.embedding_
    assert model.predict() is model.embedding_
    assert_allclose(model.predict(columns=True), expected_column)


def test_louvain_rectangular_bipartite_control():
    matrix = sparse.vstack((biadjacency(), sparse.csr_matrix((1, 3))), format='csr')
    model = LouvainEmbedding().fit(matrix)
    assert_allclose(model.embedding_, [[1.0], [0.0], [0.0], [0.0]])
    assert model.embedding_row_ is model.embedding_
    assert_allclose(model.predict(columns=True), [[1.0], [1.0], [0.0]])


def test_louvain_ordinary_square_control():
    model = LouvainEmbedding().fit(biadjacency())
    assert model.embedding_.shape[0] == 3
    assert model.embedding_row_ is None
    assert model.predict(columns=True) is None


def test_louvain_refit_clears_bipartite_outputs():
    matrix = sparse.vstack((biadjacency(), sparse.csr_matrix((1, 3))), format='csr')
    model = LouvainEmbedding().fit(matrix)
    assert model.embedding_col_ is not None
    assert model.fit(biadjacency()) is model
    assert model.embedding_.shape[0] == 3
    assert model.embedding_row_ is None
    assert model.predict(columns=True) is None
    model.fit(biadjacency(), force_bipartite=True)
    assert model.embedding_row_ is model.embedding_
    assert model.predict(columns=True).shape == (3, 1)


@pytest.mark.parametrize('which', ['remove', 'merge', 'keep'])
@pytest.mark.parametrize('square', [False, True])
def test_louvain_bipartite_singleton_modes(which, square):
    matrix = biadjacency()
    if not square:
        matrix = sparse.vstack((matrix, sparse.csr_matrix((1, 3))), format='csr')
    model = LouvainEmbedding(isolated_nodes=which).fit(matrix, force_bipartite=square)
    if which == 'remove':
        expected_row = np.array([[1.0], [0.0], [0.0]])
        expected_column = np.array([[1.0], [1.0], [0.0]])
    else:
        expected_row = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        expected_column = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    if not square:
        expected_row = np.vstack((expected_row, np.zeros(expected_row.shape[1])))
    embedding_column = model.predict(columns=True)
    assert model.embedding_.shape[1] == embedding_column.shape[1]
    actual = np.vstack((model.embedding_, embedding_column))
    expected = np.vstack((expected_row, expected_column))
    # Cluster IDs may include gaps or be permuted; the shared geometry is fixed.
    assert_allclose(actual @ actual.T, expected @ expected.T)


@pytest.mark.parametrize('which,expected_primary,expected_secondary', [
    ('remove', [0, 0, -1, -1], [-1, 0, -1, -1, -1]),
    ('merge', [0, 0, 1, 1], [-1, 0, 1, 1, -1]),
    ('keep', [2, 2, 4, 6], [-1, 2, 4, 6, -1]),
])
def test_louvain_secondary_label_mapping(which, expected_primary, expected_secondary):
    primary = np.array([2, 2, 4, 6])
    secondary = np.array([0, 2, 4, 6, 7])
    result_primary, result_secondary = reindex_labels(primary, secondary, which)
    assert_allclose(result_primary, expected_primary)
    assert_allclose(result_secondary, expected_secondary)


@pytest.mark.parametrize('which,n_components', [('remove', 0), ('merge', 1), ('keep', 3)])
def test_louvain_all_singleton_bipartite_clusters(which, n_components):
    model = LouvainEmbedding(isolated_nodes=which).fit(
        sparse.eye(3, format='csr'), force_bipartite=True)
    assert model.embedding_.shape == (3, n_components)
    assert model.predict(columns=True).shape == (3, n_components)
    if which == 'remove':
        expected = np.zeros((6, 0))
    elif which == 'merge':
        expected = np.ones((6, 1))
    else:
        expected = np.vstack((np.eye(3), np.eye(3)))
    actual = np.vstack((model.embedding_, model.predict(columns=True)))
    assert_allclose(actual @ actual.T, expected @ expected.T)


@pytest.mark.parametrize('which', ['remove', 'merge', 'keep'])
@pytest.mark.parametrize('two_clusters', [False, True])
def test_louvain_isolated_column_preserves_embedding_width(which, two_clusters):
    if two_clusters:
        matrix = sparse.hstack((biadjacency(), sparse.csr_matrix((3, 1))), format='csr')
        expected_row = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        expected_column = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        if which == 'remove':
            expected_row = expected_row[:, :1]
            expected_column = expected_column[:, :1]
        n_components = {'remove': 1, 'merge': 2, 'keep': 3}[which]
    else:
        matrix = sparse.csr_matrix([[1, 1, 0], [1, 1, 0]])
        expected_row = np.ones((2, 1))
        expected_column = np.array([[1.0], [1.0], [0.0]])
        n_components = 1 if which == 'remove' else 2
    # The complete 2-by-2 block forms one community at resolution 0.5.
    resolution = 1.0 if two_clusters else 0.5
    model = LouvainEmbedding(isolated_nodes=which, resolution=resolution).fit(matrix)
    assert model.embedding_.shape == (matrix.shape[0], n_components)
    assert model.predict(columns=True).shape == (matrix.shape[1], n_components)
    actual = np.vstack((model.embedding_, model.predict(columns=True)))
    expected = np.vstack((expected_row, expected_column))
    assert_allclose(actual @ actual.T, expected @ expected.T)
