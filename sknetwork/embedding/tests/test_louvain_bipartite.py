"""Square biadjacency inputs and refit state for Louvain embeddings."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from sknetwork.embedding import LouvainEmbedding


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
