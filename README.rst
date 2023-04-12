.. image:: https://perso.telecom-paristech.fr/bonald/logo_sknetwork.png
    :align: right
    :width: 150px
    :alt: logo sknetwork



.. image:: https://img.shields.io/pypi/v/scikit-network.svg
        :target: https://pypi.python.org/pypi/scikit-network

.. image:: https://github.com/sknetwork-team/scikit-network/actions/workflows/ci_checks.yml/badge.svg
        :target: https://github.com/sknetwork-team/scikit-network/actions/workflows/ci_checks.yml

.. image:: https://readthedocs.org/projects/scikit-network/badge/?version=latest
        :target: https://scikit-network.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://codecov.io/gh/sknetwork-team/scikit-network/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/sknetwork-team/scikit-network

.. image:: https://img.shields.io/pypi/pyversions/scikit-network.svg
        :target: https://pypi.python.org/pypi/scikit-network

Free software library in Python for machine learning on graphs:

* Memory-efficient representation of graphs as sparse matrices in scipy_ format
* Fast algorithms
* Simple API inspired by scikit-learn_

.. _scipy: https://www.scipy.org
.. _scikit-learn: https://scikit-learn.org/

Resources
---------

* Free software: BSD license
* GitHub: https://github.com/sknetwork-team/scikit-network
* Documentation: https://scikit-network.readthedocs.io

Quick start
-----------

Install scikit-network:

.. code-block:: console

    $ pip install scikit-network

Import scikit-network::

    import sknetwork

Overview
--------

An overview of the package is presented in this `notebook <https://scikit-network.readthedocs.io/en/latest/tutorials/overview/index.html>`_.

Documentation
-------------

The documentation is structured as follows:

* `Getting started <https://scikit-network.readthedocs.io/en/latest/first_steps.html>`_: First steps to install, import and use scikit-network.
* `User manual <https://scikit-network.readthedocs.io/en/latest/reference/data.html>`_: Description of each function and object of scikit-network.
* `Tutorials <https://scikit-network.readthedocs.io/en/latest/tutorials/data/index.html>`_: Application of the main tools to toy examples.
* `Examples <https://scikit-network.readthedocs.io/en/latest/use_cases/text.html>`_: Examples combining several tools on specific use cases.
* `About <https://scikit-network.readthedocs.io/en/latest/authors.html>`_: Authors, history of the library, how to contribute, index of functions and objects.

Citing
------

If you want to cite scikit-network, please refer to the publication in
the `Journal of Machine Learning Research <https://jmlr.org>`_:

.. code::

    @article{JMLR:v21:20-412,
      author  = {Thomas Bonald and Nathan de Lara and Quentin Lutz and Bertrand Charpentier},
      title   = {Scikit-network: Graph Analysis in Python},
      journal = {Journal of Machine Learning Research},
      year    = {2020},
      volume  = {21},
      number  = {185},
      pages   = {1-6},
      url     = {http://jmlr.org/papers/v21/20-412.html}
    }
