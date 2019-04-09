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


About the documentation
-----------------------

We use the following notations in the documentation:

* :math:`A` denotes the adjacency matrix for undirected and directed graphs.

* :math:`B` denotes the biadjacency matrix for bipartite graphs (possibly non-square).

* :math:`d = A1` or :math:`B1` is the out-degree vector and :math:`D = \text{diag}(d)` the associated diagonal matrix.

* :math:`f = A^T1` or :math:`B^T1` is the in-degree vector and :math:`F = \text{diag}(f)` the associated diagonal matrix.

* :math:`w = 1^T A1` or :math:`1 ^T B1` is the total weight of the graph.


