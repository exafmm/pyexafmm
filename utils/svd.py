"""
Compute low rank approximations using SVD.
"""
import numpy as np
import scipy.linalg


def compress(matrix, target_rank=None):
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

    full_rank = len(s)

    if target_rank > full_rank or target_rank is None:
        target_rank = full_rank

    for i, sv in enumerate(s):
        if i > target_rank:
            break
        else:
            uk.append(u[:, i])
            vhk.append(vh[i, :])
            sk.append(sv)

    uk = np.array(uk).T
    vhk = np.array(vhk)
    sk = np.array(sk)

    return uk, sk, vhk
