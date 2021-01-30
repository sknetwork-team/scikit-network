.. _introduction:

Getting started
***************

:mod:`scikit-network` is an open-source python package for the analysis of large graphs.


Installation
------------

Install :mod:`scikit-network`:

.. code-block:: console

    $ pip install scikit-network

Import :mod:`scikit-network` in a Python project::

    import sknetwork as skn

Data format
-----------

Each graph is represented by its :term:`adjacency` matrix, either as a dense ``numpy array``
or a sparse ``scipy CSR matrix``.
A bipartite graph can be represented by its biadjacency matrix, in the same format.

Check our tutorial Getting started for various ways of loading a graph
(from a list of edges, a dataframe or a TSV file, for instance).

Documentation
-------------

We use the following notations in the documentation:

Graphs
^^^^^^

For undirected graphs:

* :math:`A` is the adjacency matrix of the graph (dimension :math:`n\times n`)
* :math:`d = A1` is the vector of node weights (node degrees if the matrix :math:`A` is binary)
* :math:`D = \text{diag}(d)` the diagonal matrix of node weights

Digraphs
^^^^^^^^

For directed graphs:

* :math:`A` is the adjacency matrix of the graph (dimension :math:`n\times n`)
* :math:`d^+ = A1` and :math:`d^- = A^T1` are the vectors of out-weights and in-weights of nodes (out-degrees and in-degrees if the matrix :math:`A` is binary)
* :math:`D^+ = \text{diag}(d^+)` and :math:`D^- = \text{diag}(d^-)` are the diagonal matrices of out-weights and in-weights

Bigraphs
^^^^^^^^

For bipartite graphs:

* :math:`B` is the biadjacency matrix of the graph (dimension :math:`n_1\times n_2`)
* :math:`d_1 = B1` and :math:`d_2 = B^T1` are the vectors of weights (rows and columns)
* :math:`D_1 = \text{diag}(d_1)` and :math:`D_2 = \text{diag}(d_2)` are the diagonal matrices of weights.

Notes
^^^^^

* Adjacency and biadjacency matrices have non-negative entries (the weights of the edges).
* Bipartite graphs are undirected but have a special structure that is exploited by some algorithms.
  These algorithms are identified with the prefix ``Bi``.

