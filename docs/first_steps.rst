.. _getting_started:

Overview
--------

Scikit-network is an open-source python package for machine learning on graphs.

Each graph is represented by its adjacency matrix in the sparse CSR format of ``scipy``.

An overview of the package is presented in this :ref:`notebook<OverviewTag>`.

Installation
------------

To install scikit-network, run this command in your terminal:

.. code-block:: console

    $ pip install scikit-network

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

Alternately, you can download the sources from `Github`_ and run:

.. code-block:: console

    $ cd <scikit-network folder>
    $ python setup.py develop


.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/
.. _Github: https://github.com/sknetwork-team/scikit-network

Import
------

Import scikit-network in Python:

.. code-block:: python

    import sknetwork as skn

Get started
-----------

A graph is represented by its :term:`adjacency` matrix. When the graph is bipartite,
it can be represented by its :term:`biadjacency` matrix (most often, a rectangular matrix). Check our :ref:`tutorial<DataTag>` for various ways of loading a graph
(from a list of edges, a dataframe or a CSV file, for instance).

Each algorithm is represented as an object with a ``fit`` method.

Here is an example to cluster the `Karate club graph`_ with the `Louvain algorithm`_:

.. code-block:: python

    from sknetwork.data import karate_club
    from sknetwork.clustering import Louvain

    adjacency = karate_club()
    algorithm = Louvain()
    algorithm.fit(adjacency)


If the graph is bipartite, the algorithm applies to
nodes corresponding to the rows of the :term:`biadjacency` matrix; specific outputs for nodes corresponding to
rows and columns of the biadjacency matrix can be obtained with the respective suffixes `_row_` and `_col_`.

More details are provided in this :ref:`tutorial<OverviewTag>`.

.. _Karate club graph: https://en.wikipedia.org/wiki/Zachary%27s_karate_club
.. _Louvain algorithm: https://en.wikipedia.org/wiki/Louvain_method
