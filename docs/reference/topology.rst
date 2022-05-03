.. _topology:

Topology
********

Algorithms for the analysis of graph topology.

Connectivity
------------

.. autofunction:: sknetwork.topology.get_connected_components

.. autofunction:: sknetwork.topology.is_connected

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


Isomorphism
-----------

.. autoclass:: sknetwork.topology.WeisfeilerLehman

.. autofunction:: sknetwork.topology.are_isomorphic
