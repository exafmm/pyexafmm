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

    # Reconstruct approximation of matrix
    appx = None

    for i, sv in enumerate(s):
        if sv > tol:
            if appx is None:
                appx = sv*np.outer(u[i].T, vh[i])
            else:
                appx += sv*np.outer(u[i].T, vh[i])

    return appx


def randomised_compress(matrix, tol=1e-5):
    pass


if __name__ == "__main__":
    matrix = np.array([
        [1, 10, 21],
        [2, 12, 32],
    ])

    print(dense_compress(matrix))
