import pytest
from sknetwork.linalg import LanczosEig
import numpy as np
import math

def test_eigen():
	L = LanczosEig("LM")
	m = np.array([[3,2,4], [2,0, 2], [4, 2, 3]])
	L.fit(m, n_components=2)
	assert math.isclose(L.eigenvalues_[0], 8.0, rel_tol=1e-9, abs_tol=0.0)
	assert math.isclose(L.eigenvalues_[1], -1.0, rel_tol=1e-9, abs_tol=0.0)