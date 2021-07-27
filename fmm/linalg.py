"""
Linear algebra utilities
"""
import numpy as np
import scipy.linalg.lapack as lapack


EPS = {
    np.float32: np.finfo(np.float32).eps,
    np.float64: np.finfo(np.float64).eps
}


def _svd(a, full_matrices, dtype):
    """
    Type dependent Lapack routine
    """
    if dtype == np.float32:
        return lapack.sgesvd(a, full_matrices=full_matrices)
    elif dtype == np.float64:
        return lapack.dgesvd(a, full_matrices=full_matrices)


def pinv(a):
    """
    Moore-Penrose Pseudo-Inverse calculation via SVD.
    Parameters:
    -----------
    a : np.array(shape=(m, n), dtype=np.float32)
    """
    dtype = a.dtype
    u, s, vt, _ = _svd(a, full_matrices=0, dtype=dtype.type)
    max_s = max(s)

    for i in range(len(s)):
        s[i] = 1./s[i] if s[i] > 4*max_s*EPS[dtype.type] else dtype.type(0.)

    v = vt.T
    ut = u.T

    return v @ np.diag(s) @  ut


def pinv2(a):
    """
    Moore-Penrose Pseudo-Inverse calculation via SVD. Return SVD result in two
        components, as in Malhotra et. al. (2015).
    Parameters:
    -----------
    a : np.array(shape=(m, n), dtype=float)
    """
    dtype = a.dtype
    u, s, vt, _ = _svd(a, full_matrices=0, dtype=dtype.type)
    max_s = max(s)

    for i in range(len(s)):
        s[i] = 1./s[i] if s[i] > 4*max_s*EPS[dtype.type] else dtype.type(0.)

    v = vt.T
    ut = u.T

    return v @ np.diag(s),  ut
