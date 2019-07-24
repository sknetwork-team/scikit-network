.. _basics:

Basics
******

.. currentmodule:: sknetwork

This module contains well-known algorithms for shortest path computation, connectedness of the graph and searches.

Many algorithms are adapted from SciPy_.

.. automodule:: sknetwork.basics

Shortest paths
--------------

.. autofunction:: sknetwork.basics.shortest_path

Summary of the different methods and their worst-case complexity (for the all-pairs problem):

+------+----------------+--------------------------------+-------------------------------------------------+
|        Method         |    Worst-case time complexity  |                          Remarks                |
+------+----------------+                                +                                                 +
| Code |    Full name   |                                |                                                 |
+======+================+================================+=================================================+
|  FW  | Floyd-Warshall | :math:`O(|V|^3)`               | All-pairs shortest path problem only            |
+------+----------------+--------------------------------+-------------------------------------------------+
|   D  |    Dijkstra    | :math:`O(|V|^2 \log |V| + |E|)`| For use on graphs with positive weights only    |
+------+----------------+--------------------------------+-------------------------------------------------+
|  BM  |  Bellman-Ford  | :math:`O(|V||E|)`              |                                                 |
+------+----------------+--------------------------------+   | For use on graphs without                   +
|   J  |    Johnson     | :math:`O(|V|^2 \log |V| + |E|)`|   | negative-weight cycles only                 |
+------+----------------+--------------------------------+-------------------------------------------------+


Searches
--------

.. autofunction:: sknetwork.basics.breadth_first_search

.. autofunction:: sknetwork.basics.depth_first_search



Structure
---------
.. autofunction:: sknetwork.basics.is_connected

.. autofunction:: sknetwork.basics.connected_components

.. autofunction:: sknetwork.basics.largest_connected_component

.. autofunction:: sknetwork.basics.is_bipartite


.. _SciPy: https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html
