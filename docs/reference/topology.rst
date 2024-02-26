.. _topology:

Topology
********

Functions related to graph topology.

Connectivity
------------

.. autofunction:: sknetwork.topology.get_connected_components

.. autofunction:: sknetwork.topology.is_connected

.. autofunction:: sknetwork.topology.get_largest_connected_component

Structure
---------

.. autofunction:: sknetwork.topology.is_bipartite

Cycles
------

.. autofunction:: sknetwork.topology.is_acyclic

.. autofunction:: sknetwork.topology.get_cycles

.. autofunction:: sknetwork.topology.break_cycles

Core decomposition
------------------

.. autoclass:: sknetwork.topology.get_core_decomposition


Triangles
---------

.. autoclass:: sknetwork.topology.count_triangles

.. autoclass:: sknetwork.topology.get_clustering_coefficient


Cliques
-------

.. autoclass:: sknetwork.topology.count_cliques


Isomorphism
-----------

.. autoclass:: sknetwork.topology.color_weisfeiler_lehman

.. autofunction:: sknetwork.topology.are_isomorphic
