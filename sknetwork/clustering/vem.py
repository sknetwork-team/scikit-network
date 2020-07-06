#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 2020
@author: Clément Bonet <cbonet@enst.fr>
"""

from typing import Union

import numpy as np
from scipy import sparse

import numba

from sknetwork.clustering.base import BaseClustering
from sknetwork.clustering import KMeans
from sknetwork.embedding import GSVD

from copy import deepcopy


eps = np.finfo(float).eps
xmin = np.finfo(np.float).min
xmax = np.finfo(np.float).max


@numba.jit(nopython=True,parallel=True)
def J(X,taus,alphas,pis,Q):
    n = X.shape[0]

    output = np.sum(taus@np.log(alphas))

    cpt = 0
    for i in numba.prange(n):
        for j in numba.prange(n):
            if i!=j:
                for q in numba.prange(Q):
                    for l in numba.prange(Q):
                        logb = (X[i,j]*np.log(pis[q,l])+ \
                               (1-X[i,j])*np.log(1-pis[q,l]))
                        cpt += taus[i,q]*taus[j,l]*logb

    return output+cpt/2-np.sum(taus*np.log(taus))


@numba.jit(nopython=True,parallel=True)
def E_step_VEM(X,taus,alphas,pis,Q):
    n = X.shape[0]
    logTau = np.log(np.maximum(taus,eps))
    for i in numba.prange(n):
        logTau[i,:] = np.log(alphas)
        for q in numba.prange(Q):
            for j in numba.prange(n):
                if j!=i:
                    for l in numba.prange(Q):
                        logTau[i,q] += taus[j,l]*(X[i,j]*np.log(pis[q,l])+\
                                                 (1-X[i,j])*np.log(1-pis[q,l]))

    logTau = np.maximum(np.minimum(logTau,xmax),xmin)
    tau = np.exp(logTau)

    for i in numba.prange(n):
        tau[i,:] /= np.sum(tau[i,:])

    return np.maximum(tau,eps)


@numba.jit(nopython=True,parallel=True)
def M_step_VEM(X,taus,alphas,pis,Q):
    n = X.shape[0]
    alphas = np.maximum(np.sum(taus,axis=0)/n,eps)

    for q in numba.prange(Q):
        for l in range(Q):
            num = 0
            denom = 0
            for i in numba.prange(n):
                for j in numba.prange(n):
                    if i!=j:
                        num += taus[i,q]*taus[j,l]*X[i,j]
                        denom += taus[i,q]*taus[j,l]
            if denom>eps:
                pi = num/denom
            else:
                ## class with a single vertex
                pi = 0.5

            pis[q,l] = np.minimum(np.maximum(num/denom,eps),1-eps)

    return alphas


class VEM(BaseClustering):
    """ Variational EM

    Parameters
    ----------
    n_clusters:
        Number of desired clusters.
    init:
        {‘kmeans’, ‘random’}, defaults to ‘kmeans’
    max_iter:
        Maximum number of iterations to perform.
    tol:
        VEM will stop when the likelihood is not improved more than tol, defaults 1e-6
    Attributes
    ----------
    labels: np.ndarray
        Label of each node.
    membership_ : sparse.csr_matrix
        Membership matrix.
    adjacency_ : sparse.csr_matrix
        Adjacency matrix between clusters.

    Example
    -------
    >>> from sknetwork.clustering import VEM
    >>> from sknetwork.data import karate_club
    >>> vem = VEM(n_clusters=3)
    >>> adjacency = karate_club()
    >>> labels = vem.fit_transform(adjacency)
    >>> len(set(labels))
    3
    """
    def __init__(self,n_clusters : int = 3, init: str = "kmeans",
                 max_iter : int = 100,tol:float = 1e-6,sort_clusters:bool = True,
                 return_membership: bool=True, return_aggregate: bool = True):
        super(VEM, self).__init__(sort_clusters=sort_clusters,
                                  return_membership=return_membership,
                                  return_aggregate=return_aggregate)
        self.n_clusters = n_clusters

        if init!="kmeans" and init!="random":
            raise ValueError("Unknown initialization. It must be either 'kmeans' or 'random'")

        self.init = init
        self.max_iter = max_iter
        self.tol = tol

    def _init_vem(self,init):
        n = self.X.shape[0]

        if init=="kmeans":
            kmeans = KMeans(n_clusters = self.n_clusters, embedding_method=GSVD(self.n_clusters))
            labels_k = kmeans.fit_transform(self.X)

            self.taus = np.zeros(shape=(n,self.n_clusters))
            self.taus[:] = np.eye(self.n_clusters)[labels_k]
        else:
            self.taus = np.random.rand(n,self.n_clusters)
            for i in range(n):
                self.taus[i,:] /= np.sum(self.taus[i,:])

        self.alphas = np.ones(shape=(self.n_clusters,))/self.n_clusters
        self.pis = np.zeros(shape=(self.n_clusters,self.n_clusters))
        

    def _j(self):
        n = self.X.shape[0]
        return J(self.X@np.eye(n),self.taus,self.alphas,self.pis,self.n_clusters)


    def _e_step(self):
        n = self.X.shape[0]
        self.taus = E_step_VEM(self.X@np.eye(n),self.taus,self.alphas,self.pis,self.n_clusters)


    def _m_step(self):
        n = self.X.shape[0]
        self.alphas = M_step_VEM(self.X@np.eye(n),self.taus,self.alphas,self.pis,self.n_clusters)


    def fit(self,adjacency: Union[sparse.csr_matrix,np.ndarray]) -> 'VEM':
        self.X = deepcopy(adjacency)
        n = self.X.shape[0]

        self._init_vem(self.init)

        Js = []

        for k in range(self.max_iter):
            self._m_step()
            self._e_step()

            Js.append(self._j())

            if len(Js)>1 and Js[-1]-Js[-2]<self.tol:
                break


        self.labels_ = np.argmax(self.taus,axis=1)
        self._secondary_outputs(adjacency)

        return self
