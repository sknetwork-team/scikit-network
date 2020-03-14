#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar, 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""

from functools import partial
from multiprocessing import Pool
from typing import Union, Optional

import numpy as np
from scipy import sparse

from sknetwork.classification import BaseClassifier
from sknetwork.ranking import BaseRanking, BiDiffusion, BiPageRank, Diffusion, PageRank
from sknetwork.utils.checks import check_seeds, check_labels
from sknetwork.utils.verbose import VerboseMixin


class RankClassifier(BaseClassifier, VerboseMixin):
    """Generic class for ranking based classifiers.

    Parameters
    ----------
    algorithm:
        Which ranking algorithm to use.
    n_jobs:
        If an integer value is given, denotes the number of workers to use (-1 means the maximum number will be used).
        If ``None``, no parallel computations are made.

    """

    def __init__(self, algorithm: BaseRanking, n_jobs: Optional[int] = None, verbose: bool = False):
        super(RankClassifier, self).__init__()
        VerboseMixin.__init__(self, verbose)

        self.algorithm = algorithm
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.membership_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], seeds: Union[np.ndarray, dict]):
        """

        Parameters
        ----------
        adjacency
        seeds

        Returns
        -------

        """
        seeds_labels = check_seeds(seeds, adjacency)
        classes, n_classes = check_labels(seeds_labels)
        n: int = adjacency.shape[0]

        personalizations = []
        if isinstance(self.algorithm, Diffusion):
            for label in classes:
                personalization = -np.ones(n)
                personalization[seeds_labels == label] = 1
                ix = np.logical_and(seeds_labels != label, seeds_labels >= 0)
                personalization[ix] = 0
                personalizations.append(personalization)
        else:
            for label in classes:
                personalization = np.array(seeds_labels == label).astype(int)
                personalizations.append(personalization)

        if self.n_jobs != 1:
            local_function = partial(self.algorithm.fit_transform, adjacency)
            with Pool(self.n_jobs) as pool:
                membership = np.array(pool.map(local_function, personalizations))
            membership = membership.T
        else:
            membership = np.zeros((n, n_classes))
            for i in range(n_classes):
                membership[:, i] = self.algorithm.fit_transform(adjacency, personalization=personalizations[i])[:n]

        if isinstance(self.algorithm, Diffusion):
            membership -= np.mean(membership, axis=0)
            membership = np.exp(membership)

        norms = membership.sum(axis=1)
        ix = np.argwhere(norms == 0).ravel()
        if len(ix) > 0:
            self.log.print('Nodes ', ix, ' have a null membership.')
        membership[~ix] /= norms[~ix, np.newaxis]

        labels = np.argmax(membership, axis=1)
        self.labels_ = np.array([classes[val] for val in labels]).astype(int)
        self.membership_ = membership

        return self


class MaxRank(RankClassifier):
    """Semi-supervised node classification using multiple personalized PageRanks.

    Parameters
    ----------
    damping_factor:
        Damping factor for personalized PageRank.
    solver : str
        Which solver to use: 'spsolve', 'lanczos' (default), 'lsqr' or 'halko'.
        Otherwise, the random walk is emulated for a certain number of iterations.
    n_iter : int
        If ``solver`` is not one of the standard values, the pagerank is approximated by emulating the random walk for
        ``n_iter`` iterations.

    Example
    -------
    >>> from sknetwork.data import karate_club
    >>> maxrank = MaxRank()
    >>> adjacency, labels_true = karate_club(return_labels=True)
    >>> seeds = {0: labels_true[0], 33: labels_true[33]}
    >>> labels_pred = maxrank.fit_transform(adjacency, seeds)
    >>> np.round(np.mean(labels_pred == labels_true), 2)
    0.97

    References
    ----------
    Lin, F., & Cohen, W. W. (2010, August). `Semi-supervised classification of network data using very few labels.
    <https://lti.cs.cmu.edu/sites/default/files/research/reports/2009/cmulti09017.pdf>`_
    In 2010 International Conference on Advances in Social Networks Analysis and Mining (pp. 192-199). IEEE.

    """

    def __init__(self, damping_factor: float = 0.85, solver: str = 'bicgstab', n_iter: int = 10,
                 n_jobs: Optional[int] = None, verbose: bool = False):
        algorithm = PageRank(damping_factor, solver, n_iter)
        super(MaxRank, self).__init__(algorithm, n_jobs, verbose)


class BiMaxRank(RankClassifier):
    """Semi-supervised node classification using multiple personalized PageRanks for bipartite graphs.

    Parameters
    ----------
    damping_factor:
        Damping factor for personalized PageRank.
    solver : str
        Which solver to use: 'spsolve', 'lanczos' (default), 'lsqr' or 'halko'.
        Otherwise, the random walk is emulated for a certain number of iterations.
    n_iter : int
        If ``solver`` is not one of the standard values, the pagerank is approximated by emulating the random walk for
        ``n_iter`` iterations.

    """

    def __init__(self, damping_factor: float = 0.85, solver: str = None, n_iter: int = 10,
                 n_jobs: Optional[int] = None, verbose: bool = False):
        algorithm = BiPageRank(damping_factor, solver, n_iter)
        super(BiMaxRank, self).__init__(algorithm, n_jobs, verbose)


class MaxDiff(RankClassifier):
    """Semi-supervised node classification using multiple diffusions.

    Parameters
    ----------
    n_iter: int
        If ``n_iter > 0``, the algorithm will emulate the diffusion for n_iter steps.
        If ``n_iter <= 0``, the algorithm will use BIConjugate Gradient STABilized iteration
        to solve the Dirichlet problem.

    Example
    -------
    >>> from sknetwork.data import karate_club
    >>> maxdiff = MaxDiff()
    >>> adjacency, labels_true = karate_club(return_labels=True)
    >>> seeds = {0: labels_true[0], 33: labels_true[33]}
    >>> labels_pred = maxdiff.fit_transform(adjacency, seeds)
    >>> np.round(np.mean(labels_pred == labels_true), 2)
    0.97

    References
    ----------
    Lin, F., & Cohen, W. W. (2010, August). `Semi-supervised classification of network data using very few labels.
    <https://lti.cs.cmu.edu/sites/default/files/research/reports/2009/cmulti09017.pdf>`_
    In 2010 International Conference on Advances in Social Networks Analysis and Mining (pp. 192-199). IEEE.

    """
    def __init__(self, n_iter: int = 10, n_jobs: Optional[int] = None, verbose: bool = False):
        algorithm = Diffusion(n_iter, verbose)
        super(MaxDiff, self).__init__(algorithm, n_jobs, verbose)


class BiMaxDiff(RankClassifier):
    """Semi-supervised node classification using multiple diffusions.

    Parameters
    ----------
    n_iter: int
        If ``n_iter > 0``, the algorithm will emulate the diffusion for n_iter steps.
        If ``n_iter <= 0``, the algorithm will use BIConjugate Gradient STABilized iteration
        to solve the Dirichlet problem.

    Example
    -------
    >>> from sknetwork.data import karate_club
    >>> maxdiff = MaxDiff()
    >>> adjacency, labels_true = karate_club(return_labels=True)
    >>> seeds = {0: labels_true[0], 33: labels_true[33]}
    >>> labels_pred = maxdiff.fit_transform(adjacency, seeds)
    >>> np.round(np.mean(labels_pred == labels_true), 2)
    0.97

    References
    ----------
    Lin, F., & Cohen, W. W. (2010, August). `Semi-supervised classification of network data using very few labels.
    <https://lti.cs.cmu.edu/sites/default/files/research/reports/2009/cmulti09017.pdf>`_
    In 2010 International Conference on Advances in Social Networks Analysis and Mining (pp. 192-199). IEEE.

    """
    def __init__(self, n_iter: int = 10, n_jobs: Optional[int] = None, verbose: bool = False):
        algorithm = BiDiffusion(n_iter, verbose)
        super(BiMaxDiff, self).__init__(algorithm, n_jobs, verbose)
