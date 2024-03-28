#!/usr/bin/env python3
# coding: utf-8
"""
Created in April 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""
from typing import Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.gnn.activation import BaseActivation
from sknetwork.gnn.loss import BaseLoss
from sknetwork.gnn.base_layer import BaseLayer
from sknetwork.utils.check import add_self_loops
from sknetwork.linalg import diagonal_pseudo_inverse


class Convolution(BaseLayer):
    """Graph convolutional layer.

    Apply the following function to the embedding :math:`X`:

    :math:`\\sigma(\\bar AXW + b)`,

    where :math:`\\bar A` is the normalized adjacency matrix (possibly with inserted self-embeddings),
    :math:`W`, :math:`b` are trainable parameters and :math:`\\sigma` is the activation function.

    Parameters
    ----------
    layer_type : str
        Layer type. Can be either ``'Conv'``, convolutional operator as in [1] or ``'Sage'``, as in [2].
    out_channels: int
        Dimension of the output.
    activation: str (default = ``'Relu'``) or custom activation.
        Activation function.
        If a string, can be either ``'Identity'``, ``'Relu'``, ``'Sigmoid'`` or ``'Softmax'``.
    use_bias: bool (default = `True`)
        If ``True``, add a bias vector.
    normalization: str (default = ``'both'``)
        Normalization of the adjacency matrix for message passing.
        Can be either `'left'`` (left normalization by the degrees), ``'right'`` (right normalization by the degrees),
        ``'both'`` (symmetric normalization by the square root of degrees, default) or ``None`` (no normalization).
    self_embeddings: bool (default = `True`)
        If ``True``, consider self-embedding in addition to neighbors embedding for each node of the graph.
    sample_size: int (default = 25)
        Size of neighborhood sampled for each node. Used only for ``'Sage'`` layer.

    Attributes
    ----------
    weight: np.ndarray,
        Trainable weight matrix.
    bias: np.ndarray
        Bias vector.
    embedding: np.ndarray
        Embedding of the nodes (before activation).
    output: np.ndarray
        Output of the layer (after activation).

    References
    ----------
    [1] Kipf, T., & Welling, M. (2017).
    `Semi-supervised Classification with Graph Convolutional Networks.
    <https://arxiv.org/pdf/1609.02907.pdf>`_
    5th International Conference on Learning Representations.

    [2] Hamilton, W. Ying, R., & Leskovec, J. (2017)
    `Inductive Representation Learning on Large Graphs.
    <https://arxiv.org/pdf/1706.02216.pdf>`_
    NIPS
    """
    def __init__(self, layer_type: str, out_channels: int, activation: Optional[Union[BaseActivation, str]] = 'Relu',
                 use_bias: bool = True, normalization: str = 'both', self_embeddings: bool = True,
                 sample_size: int = None, loss: Optional[Union[BaseLoss, str]] = None):
        super(Convolution, self).__init__(layer_type, out_channels, activation, use_bias, normalization,
                                          self_embeddings, sample_size, loss)

    def forward(self, adjacency: Union[sparse.csr_matrix, np.ndarray],
                features: Union[sparse.csr_matrix, np.ndarray]) -> np.ndarray:
        """Compute graph convolution.

        Parameters
        ----------
        adjacency
            Adjacency matrix of the graph.
        features : sparse.csr_matrix, np.ndarray
            Input feature of shape :math:`(n, d)` with :math:`n` the number of nodes in the graph and :math:`d`
            the size of feature space.

        Returns
        -------
        output: np.ndarray
            Output of the layer.
        """
        if not self.weights_initialized:
            self._initialize_weights(features.shape[1])

        n_row, n_col = adjacency.shape

        weights = adjacency.dot(np.ones(n_col))
        if self.normalization == 'left':
            d_inv = diagonal_pseudo_inverse(weights)
            adjacency = d_inv.dot(adjacency)
        elif self.normalization == 'right':
            d_inv = diagonal_pseudo_inverse(weights)
            adjacency = adjacency.dot(d_inv)
        elif self.normalization == 'both':
            d_inv = diagonal_pseudo_inverse(np.sqrt(weights))
            adjacency = d_inv.dot(adjacency).dot(d_inv)

        if self.self_embeddings:
            adjacency = add_self_loops(adjacency)

        message = adjacency.dot(features)
        embedding = message.dot(self.weight)

        if self.use_bias:
            embedding += self.bias

        output = self.activation.output(embedding)

        self.embedding = embedding
        self.output = output

        return output


def get_layer(layer: Union[BaseLayer, str] = 'conv', **kwargs) -> BaseLayer:
    """Get layer.

    Parameters
    ----------
    layer : str or custom layer
        If a string, must be either ``'Conv'`` (Convolution) or ``'Sage'`` (GraphSAGE).

    Returns
    -------
    Layer object.
    """
    if issubclass(type(layer), BaseLayer):
        return layer
    elif type(layer) == str:
        layer = layer.lower()
        if 'sage' in layer:
            kwargs['normalization'] = 'left'
            kwargs['self_embeddings'] = True
            return Convolution('sage', **kwargs)
        elif 'conv' in layer:
            return Convolution('conv', **kwargs)
        else:
            raise ValueError("Layer name must be \"Conv\" or \"Sage\".")
    else:
        raise TypeError("Layer must be a string or a \"BaseLayer\" object.")
