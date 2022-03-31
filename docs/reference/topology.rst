.. _topology:

Topology
********

Algorithms for the analysis of graph topology.

Connectivity
------------

.. autofunction:: sknetwork.topology.get_connected_components

.. autofunction:: sknetwork.topology.get_largest_connected_component

Structure
---------

.. autofunction:: sknetwork.topology.is_bipartite

.. autofunction:: sknetwork.topology.is_acyclic

.. autoclass:: sknetwork.topology.DAG

Core decomposition
------------------

.. autoclass:: sknetwork.topology.CoreDecomposition


Cliques
-------

.. autoclass:: sknetwork.topology.Triangles

.. autoclass:: sknetwork.topology.Cliques

Coloring
--------

.. autoclass:: sknetwork.topology.WeisfeilerLehman

Similarity
----------

.. autofunction:: sknetwork.topology.are_isomorphic
