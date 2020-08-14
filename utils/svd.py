"""
Compute low rank approximations using SVD.
"""
import numpy as np
import scipy.linalg


def compress(matrix, tol=0):
    """
    Create a low rank approximation of a matrix by cutting off terms in the SVD
        sum whose singular values are lower than a specified tolerance.

    Parameters:
    -----------
    matrix : np.array(shape=(n, m))
    tol : np.float64

    Returns:
    --------
    np.array(shape=(n, m))
        Low rank approximation of input matrix.
    """
    u, s, vh = scipy.linalg.svd(matrix)

    uk = []
    vhk = []
    sk = []

    for i, sv in enumerate(s):
        if sv < tol:
            break
        else:
            uk.append(u[:, i])
            vhk.append(vh[i, :])
            sk.append(sv)

    uk = np.array(uk).T
    vhk = np.array(vhk)
    sk = np.array(sk)

    return uk, sk, vhk
