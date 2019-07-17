.. _introduction:

Introduction
************

scikit-network is an open-source python package for the analysis of large graphs.
Graphs are represented by their adjacency matrix, in one of the following two formats:

* numpy array
* scipy sparse csr matrix

Quick start
-----------

Install scikit-network:

.. code-block:: console

    $ pip install scikit-network

To use scikit-network in a project::

    import sknetwork as skn


Optional dependency
-------------------

Numba_ is not listed as an explicit dependency for scikit-network. However it is automatically
used whenever possible -  unless specified otherwise by the user - in order to speed algorithm executions up.

Note that when using a Numba-accelerated function for the first time in a session,
just-in-time compilation takes a few seconds.

About the documentation
-----------------------

We use the following notations in the documentation:

Undirected graphs
^^^^^^^^^^^^^^^^^

* :math:`A` is the adjacency matrix of the graph (dimension :math:`n\times n`)
* :math:`d = A1` is the vector of node weights (node degrees if the matrix :math:`A` is binary)
* :math:`D = \text{diag}(d)` the diagonal matrix of node weights
* :math:`L = D - A` is the Laplacian matrix of the graph
* :math:`P = D^{-1}A` is the transition matrix of the random walk in the graph (for positive node weights)
* :math:`w = 1^T A1` is the total weight of the graph (total weight of nodes)

Directed graphs
^^^^^^^^^^^^^^^

* :math:`A` is the adjacency matrix of the graph (dimension :math:`n\times n`)
* :math:`d^+ = A1` is the vector of node out-weights (node out-degrees if the matrix :math:`A` is binary)
* :math:`D^+ = \text{diag}(d^+)` is the diagonal matrix of node out-weights
* :math:`P^+= {D^+}^{-1}A` is the transition matrix of the random walk in the graph (for positive node out-weights)
* :math:`d^- = A1` is the vector of node in-weights (node in-degrees if the matrix :math:`A` is binary)
* :math:`D^- = \text{diag}(d^-)` is the diagonal matrix of node in-weights
* :math:`P^-= {D^-}^{-1}A` is the transition matrix of the random walk in reverse direction (for positive node in-weights)
* :math:`w = 1^T A1` is the total weight of the graph (total weight of nodes)

Bipartite graphs
^^^^^^^^^^^^^^^^

* :math:`B` denotes the biadjacency matrix of the graph (dimension :math:`n\times p`); rows and columns are viewed as samples and features
* :math:`d = B1` is the vector of sample weights
* :math:`D = \text{diag}(d)` is the diagonal matrix of sample weights
* :math:`f = B^T1` is the vector of feature weights
* :math:`F = \text{diag}(f)` the diagonal matrix of feature weights
* :math:`w = 1^T B1` is the total weight of the graph (total weight of edges)

Notes
^^^^^

* Adjacency and biadjacency matrices have non-negative entries (the edges of the graph)
* Nodes are indexed from 1 to :math:`n` in the documentation and from 0 to :math:`n-1` in the code
* Bipartite graphs are undirected but have a special structure that is exploited by some algorithms

.. _Numba: https://numba.pydata.org

