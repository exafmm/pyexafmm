"""
Linear algebra utilities
"""
import numba
import numpy as np


@numba.njit(cache=True)
def pinv(a, tol=1e-4):
    """
    Moore-Penrose Pseudo-Inverse calculation via SVD.

    Parameters:
    -----------
    a : np.array(shape=(m, n), dtype=np.float32)
    tol : np.float32
        Cutoff singular values greater than 4*max(s)*tol, default 1e-4 works
        well in practice.
    """
    u, s, vt = np.linalg.svd(a, full_matrices=False)

    max_s = max(s)

    for i in range(len(s)):
        s[i] = 1./s[i] if s[i] > 4*max_s*tol else 0.

    v = vt.T
    ut = u.T

    return v @ np.diag(s) @  ut