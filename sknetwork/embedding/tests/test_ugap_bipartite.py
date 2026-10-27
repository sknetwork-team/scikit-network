"""UGAP output conventions for bipartite inputs and repeated fits."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from sknetwork.data import movie_actor
from sknetwork.embedding import UGAP
from sknetwork.utils.format import bipartite2undirected


@pytest.mark.parametrize('empty', [False, True])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('dense', [False, True])
def test_ugap_bipartite_outputs(empty, transpose, dense):
    biadjacency = sparse.csr_matrix((2, 3)) if empty else movie_actor()
    if transpose:
        biadjacency = biadjacency.T.tocsr()
    input_matrix = biadjacency.toarray() if dense else biadjacency
    original = input_matrix.copy()
    parameters = dict(n_epochs=5, n_jobs=1, random_state=42)
    model = UGAP(**parameters)
    embedding = model.fit_transform(input_matrix)
    n_row, n_col = biadjacency.shape

    assert embedding.shape == (n_row, 2)
    assert model.embedding_row_ is embedding
    assert model.transform() is embedding
    assert model.predict() is embedding
    assert model.predict(columns=True).shape == (n_col, 2)

    adjacency = bipartite2undirected(biadjacency)
    expected = UGAP(**parameters).fit_transform(adjacency)
    assert np.isfinite(embedding).all()
    assert np.isfinite(model.embedding_col_).all()
    assert_allclose(embedding, expected[:n_row], rtol=0, atol=0)
    assert_allclose(model.embedding_col_, expected[n_row:], rtol=0, atol=0)
    if dense:
        assert_array_equal(input_matrix, original)
    else:
        assert_array_equal(input_matrix.data, original.data)
        assert_array_equal(input_matrix.indices, original.indices)
        assert_array_equal(input_matrix.indptr, original.indptr)


def test_ugap_refit_clears_bipartite_outputs():
    model = UGAP(n_components=3, n_epochs=5, n_jobs=1, random_state=42)
    for adjacency in [sparse.eye(5, format='csr'), sparse.csr_matrix((1, 1))]:
        model.fit(movie_actor())
        assert model.embedding_row_ is model.embedding_
        assert model.embedding_col_ is not None
        assert model.fit(adjacency) is model
        assert model.embedding_.shape == (adjacency.shape[0], 3)
        assert model.embedding_row_ is None
        assert model.predict(columns=True) is None
