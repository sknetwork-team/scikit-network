cimport cython
from libc.stdlib cimport rand, srand
from libc.math cimport pow, fmax
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)

cpdef sgd(int n_components, int n_epochs, int n, int[:] row, int[:] col,
            double[:,:] low_dim, float a, float b, float lr,
            int negative_sampling_rate, double[:] epochs_per_sample, double[:] epoch_of_next_sample,
            unsigned int seed=0, float gamma=1.0):

    if low_dim.shape[0] != n or low_dim.shape[1] != n_components:
        raise ValueError('Embedding shape must match n and n_components.')
    srand(seed)
    cdef int num_samples = epochs_per_sample.shape[0]
    cdef int epoch, idx, i, j, k
    cdef double alpha, temp, coef, grad, d
    cdef double[::1] delta = np.empty(n_components, dtype=np.float64)

    for epoch in range(n_epochs):
        alpha = 1.0 - float(epoch) / n_epochs
        for idx in range(num_samples):
            if epoch >= epoch_of_next_sample[idx]:
                i = row[idx]
                j = col[idx]
                d = 0.0
                for k in range(n_components):
                    delta[k] = low_dim[i,k] - low_dim[j,k]
                    d += delta[k] * delta[k]

                temp = pow(d + 1e-6, b - 1)
                coef = -2.0*a*b*temp/ (1.0 + a * d * temp)

                for k in range(n_components):
                    grad = coef * delta[k]
                    low_dim[i,k] += lr * alpha * grad
                    low_dim[j,k] -= lr * alpha * grad

                for _ in range(negative_sampling_rate):
                    j = rand() % n
                    d = 0.0
                    for k in range(n_components):
                        delta[k] = low_dim[i,k] - low_dim[j,k]
                        d += delta[k] * delta[k]

                    temp = pow(d + 1e-6, b - 1)
                    coef = 2.0*gamma*b / ((0.001 + d) * (1.0 + a * d * temp))

                    for k in range(n_components):
                        grad = coef * delta[k]
                        low_dim[i,k] += lr * alpha * grad

                epoch_of_next_sample[idx] += epochs_per_sample[idx]

    return low_dim