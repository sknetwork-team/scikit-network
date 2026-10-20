"""Weighted F1 uses the same valid samples for scores and class supports."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse

from sknetwork.classification import DiffusionClassifier, get_average_f1_score


@pytest.mark.parametrize('labels_true,labels_pred,expected', [
    ([0, 1, 1], [0, 0, -2], 1 / 3),
    ([0, 1, 2, 2], [0, 0, 1, -1], 2 / 9),
    ([0, 1, 1, -1, 2], [0, 0, 1, 2, -1], 2 / 3),
    ([0, 1, 1], [0, 0, 1], 2 / 3),
    ([-1, 0, 1, 1], [1, 0, 0, 1], 2 / 3),
    ([0, 0, 1, 1], [0, -1, 1, 1], 1.0),
])
def test_weighted_f1_with_ignored_samples(labels_true, labels_pred, expected):
    labels_true, labels_pred = np.array(labels_true), np.array(labels_pred)
    original_true, original_pred = labels_true.copy(), labels_pred.copy()
    actual = get_average_f1_score(labels_true, labels_pred, average='weighted')
    assert actual == pytest.approx(expected)
    mask = (labels_true >= 0) & (labels_pred >= 0)
    filtered = get_average_f1_score(labels_true[mask], labels_pred[mask], average='weighted')
    assert actual == pytest.approx(filtered)
    assert_array_equal(labels_true, original_true)
    assert_array_equal(labels_pred, original_pred)


def test_weighted_f1_with_diffusion_abstentions():
    path = sparse.diags([1.0, 1.0], [-1, 1], shape=(4, 4), format='csr')
    edge = sparse.csr_matrix([[0, 1], [1, 0]])
    adjacency = sparse.block_diag((path, edge), format='csr')
    labels_true = np.array([0, 1, 1, 1, 0, 0])
    labels_pred = DiffusionClassifier().fit_predict(adjacency, {0: 0, 3: 1})
    assert_array_equal(labels_pred, [0, 0, 1, 1, -1, -1])
    # Retained supports are 1 and 3; the F1 scores are 2/3 and 4/5.
    expected = (2 / 3 + 3 * 4 / 5) / 4
    actual = get_average_f1_score(labels_true, labels_pred, average='weighted')
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize('labels_true,labels_pred', [
    ([-1, -2], [0, 1]),
    ([0, 1], [-1, -2]),
])
def test_weighted_f1_without_jointly_valid_samples(labels_true, labels_pred):
    with pytest.raises(ValueError, match='No sample'):
        get_average_f1_score(np.array(labels_true), np.array(labels_pred), average='weighted')
