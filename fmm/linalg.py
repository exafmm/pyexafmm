"""
Linear algebra utilities
"""
import numpy as np
import scipy.linalg.lapack as lapack

EPS = np.finfo(np.float32).eps


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
    u, s, vt, _ = lapack.sgesvd(a, full_matrices=0)
    max_s = max(s)

    for i in range(len(s)):
        s[i] = 1./s[i] if s[i] > 4*max_s*tol else 0.

    v = vt.T
    ut = u.T

    return v @ np.diag(s) @  ut


def pinv2(a):
    """
    Moore-Penrose Pseudo-Inverse calculation via SVD. Return SVD result in two
        components, as in Malhotra et. al. (2015).

    Parameters:
    -----------
    a : np.array(shape=(m, n), dtype=np.float32)
    tol : np.float32
        Cutoff singular values greater than 4*max(s)*EPS
    """
    u, s, vt, _ = lapack.sgesvd(a, full_matrices=0)
    max_s = max(s)

    for i in range(len(s)):
        s[i] = 1./s[i] if s[i] > 4*max_s*EPS else 0.

    v = vt.T
    ut = u.T

    return v @ np.diag(s),  ut

