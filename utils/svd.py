"""
Compute low rank approximations using SVD.
"""
import numpy as np
import scipy.linalg


def dense_compress(matrix, tol=1e-5):
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

    sort_idxs = np.argsort(s)

    # Reconstruct approximation of matrix
    appx = None

    for i, sv in enumerate(s[sort_idxs]):
        if sv < tol:
            pass
        else:
            if appx is None:
                appx = sv*np.outer(u[sort_idxs][i].T, vh[sort_idxs][i])
            else:
                appx += sv*np.outer(u[sort_idxs][i].T, vh[sort_idxs][i])

    return appx


def randomised_compress(matrix, tol=1e-5):
    pass


if __name__ == "__main__":
    matrix = np.array([
        [1, 10, 21],
        [2, 12, 32],
    ])

    print(dense_compress(matrix))
