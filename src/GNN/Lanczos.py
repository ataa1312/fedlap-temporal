from copy import deepcopy

import torch
import numpy as np
from scipy.linalg import schur
from tqdm import tqdm

from src import *


def estimate_eigh(A, m, X=None, method="lanczos", p=5, log=True):
    if method == "lanczos":
        T, V = Lanczos_func(A, m, log=log)
    elif method == "arnoldi":
        T, V = arnoldi_iteration(A, m, log=log)
    elif method == "block_lanczos":
        T, V = block_lanczos(A, X, m=m, p=p, log=log)
    elif method == "Randomized":
        D, U2 = randomized_eig(A, m)
        return D, U2, None

    D, U = torch.linalg.eigh(T)

    # Tj = torch.diagonal(T, offset=-1)
    # plt.plot(Tj)
    # plt.show()

    # T = T.float()
    # V = V.float()
    # D = D.float()
    # U = U.float()
    # T_inv = U @ torch.diag(1 / D) @ U.T
    # V_t = A @ V @ T_inv
    # U_t = torch.matmul(V_t, U)
    # A_t = V_t @ T @ V_t.T

    # d = torch.dist(A_t, A)
    # d2 = torch.mean(torch.abs(A.to_dense() - A_t))
    # d3 = torch.mean(torch.abs(V - V_t))

    # D_, U_ = schur(T)
    # D_ = torch.tensor(D_, dtype=torch.double, device=dev)
    # U_ = torch.tensor(U_, dtype=torch.double, device=dev)
    # D_ = D_.double()
    # U_ = U_.double()

    U2 = V @ U
    U2 = U2.float()
    D = D.float()

    # return T.float(), V.float()
    V = V.float()
    return D, U2, V


# dev = "cpu"


def Lanczos_func(A, m=10, v=None, log=True):
    n = A.shape[0]
    A = deepcopy(A).double()
    A = A.to_sparse()
    B = torch.zeros(m - 1, dtype=torch.double, device=dev)
    a = torch.zeros(m, dtype=torch.double, device=dev)
    V = torch.zeros((n, m), dtype=torch.double, device=dev)
    if v is None:
        # v = torch.ones(A.shape[0], dtype=torch.double, device=dev)
        v = torch.randn(n, dtype=torch.double, device=dev)
    v = v / torch.norm(v)
    V[:, 0] = v
    wp = torch.matmul(A, V[:, 0]).to_dense()
    # wp = torch.einsum("ij,j->i", A, V[:, 0])
    a[0] = torch.einsum("i,i", wp, V[:, 0])
    w = wp - a[0] * V[:, 0]
    if log:
        bar = tqdm(total=m - 1)
    for j in range(1, m):
        B[j - 1] = torch.norm(w)
        if B[j - 1] != 0:
            V[:, j] = w / B[j - 1]
        else:
            break
        wp = torch.matmul(A, V[:, j]).to_dense()
        # wp = torch.einsum("ij,j->i", A, V[:, j])
        a[j] = torch.einsum("i,i", wp, V[:, j])
        # a[j] = torch.einsum("i,i", wp - B[j - 1] * V[:, j - 1], V[:, j])
        w = wp - a[j] * V[:, j] - B[j - 1] * V[:, j - 1]

        # w -= V @ V.T @ w

        # T = torch.diag(a) + torch.diag(B, 1) + torch.diag(B, -1)

        # At = torch.matmul(V, torch.matmul(T, V.T))
        # e = torch.mean(torch.abs(A - At)).item()
        # bar.set_postfix({"e": e})
        if log:
            bar.update()

    T = torch.diag(a) + torch.diag(B, 1) + torch.diag(B, -1)

    return T, V


def arnoldi_iteration(A, m: int, b=None, log=True):
    # Run the Krylov iteration on the GPU when on CUDA (the sparse matvec + O(m^2)
    # Gram-Schmidt dominate; on CPU they pin the run at ~90s/snapshot for 35k-node
    # graphs). MPS sparse ops are unsupported, so fall back to CPU there.
    local_dev = dev if str(dev).startswith("cuda") else "cpu"
    """Compute a basis of the (n + 1)-Krylov subspace of the matrix A.

    This is the space spanned by the vectors {b, Ab, ..., A^n b}.

    Parameters
    ----------
    A : array_like
        An m × m array.
    b : array_like
        Initial vector (length m).
    n : int
        One less than the dimension of the Krylov subspace, or equivalently the *degree* of the Krylov space. Must be >= 1.

    Returns
    -------
    Q : numpy.array
        An m x (n + 1) array, where the columns are an orthonormal basis of the Krylov subspace.
    h : numpy.array
        An (n + 1) x n array. A on basis Q. It is upper Hessenberg.
    """
    A = deepcopy(A).double()
    A = A.to_sparse().to(local_dev)
    if b is None:
        # b = torch.ones(A.shape[0], dtype=torch.double, device=dev)  # =Laplacian null vector -> breaks down
        if config["spectral"]["deterministic_start"]:
            # fixed-seed CPU start -> identical every solve, so recompute's eigenvectors move
            # only as the graph moves (stable gauge) instead of re-randomizing each snapshot.
            g = torch.Generator().manual_seed(0)
            b = torch.randn(A.shape[0], generator=g, dtype=torch.double).to(local_dev)
        else:
            b = torch.randn(A.shape[0], dtype=torch.double, device=local_dev)
        if torch.sum(b) < 0:
            b = -b
    eps = 1e-12
    h = torch.zeros((m, m), dtype=torch.double, device=local_dev)
    Q = torch.zeros((A.shape[0], m), dtype=torch.double, device=local_dev)
    # Normalize the input vector
    Q[:, 0] = b / torch.norm(b, 2)  # Use it as the first Krylov vector
    if log:
        bar = tqdm(total=m - 1)
    for k in range(1, m):
        v = A @ Q[:, k - 1]  # Generate a new candidate vector
        for j in range(k):  # Subtract the projections on previous vectors
            h[j, k - 1] = Q[:, j].conj() @ v
            v = v - h[j, k - 1] * Q[:, j]

        h[k, k - 1] = torch.norm(v, 2)
        if h[k, k - 1] > eps:  # Add the produced vector to the list, unless
            Q[:, k] = v / h[k, k - 1]
        else:  # If that happens, stop iterating.
            return h, Q
        if log:
            bar.update()

    h = h.to(dev)
    Q = Q.to(dev)
    return h, Q


def arnoldi_iteration2(A, m: int, b=None, log=True):
    """Compute a basis of the (n + 1)-Krylov subspace of the matrix A.

    This is the space spanned by the vectors {b, Ab, ..., A^n b}.

    Parameters
    ----------
    A : array_like
        An m × m array.
    b : array_like
        Initial vector (length m).
    n : int
        One less than the dimension of the Krylov subspace, or equivalently the *degree* of the Krylov space. Must be >= 1.

    Returns
    -------
    Q : numpy.array
        An m x (n + 1) array, where the columns are an orthonormal basis of the Krylov subspace.
    h : numpy.array
        An (n + 1) x n array. A on basis Q. It is upper Hessenberg.
    """
    A = deepcopy(A).double()
    A = A.to_dense().to(dev)
    if b is None:
        b = torch.ones(A.shape[0], dtype=torch.double, device=dev)
    # b = torch.randn(A.shape[0], dtype=torch.double, device=dev)
    eps = 1e-12
    h = torch.zeros((m, m), dtype=torch.double, device=dev)
    Q = torch.zeros((A.shape[0], m), dtype=torch.double, device=dev)
    # Normalize the input vector
    Q[:, 0] = b / torch.norm(b, 2)  # Use it as the first Krylov vector
    if log:
        bar = tqdm(total=m - 1)
    for k in range(1, m):
        # v = torch.linalg.solve(A, Q[:, k - 1])
        # v = torch.linalg.inv(A.T @ A) @ A.T @ Q[:, k - 1]
        LU, pivots = torch.linalg.lu_factor(A)
        v = torch.lu_solve(Q[:, k - 1], LU, pivots)
        # v = torch.matmul(A, Q[:, k - 1]).to_dense()  # Generate a new candidate vector
        # v = torch.matmul(A, Q[:, k - 1]).to_dense()  # Generate a new candidate vector
        for j in range(k):  # Subtract the projections on previous vectors
            h[j, k - 1] = Q[:, j].conj() @ v
            v = v - h[j, k - 1] * Q[:, j]

        h[k, k - 1] = torch.norm(v, 2)
        if h[k, k - 1] > eps:  # Add the produced vector to the list, unless
            Q[:, k] = v / h[k, k - 1]
        else:  # If that happens, stop iterating.
            return h, Q
        if log:
            bar.update()
    return h, Q


def block_lanczos(H, X=None, m=10, p=5, log=True):
    n = H.shape[0]
    H = deepcopy(H).double()
    X = deepcopy(X).double()
    B = []
    A = []
    if X is None:
        X = torch.randn((n, p), dtype=torch.double, device=dev)
    else:
        p = X.shape[1]
    Qj, _ = torch.linalg.qr(X)
    Q = [Qj]
    if log:
        bar = tqdm(total=m)
    for j in range(m):
        if j == 0:
            U = H @ Q[-1]
        else:
            U = H @ Q[-1] - Q[-2] @ B[-1].T

        Aj = Q[-1].T @ U
        A.append(Aj)
        R = U - Q[-1] @ Aj
        Qj, Bj = torch.qr(R)
        Q.append(Qj)
        B.append(Bj)

        if log:
            bar.update()

    T = (
        torch.block_diag(*A)
        + torch.vstack(
            (
                torch.hstack(
                    (
                        torch.zeros((j * p, p), dtype=torch.double),
                        torch.block_diag(*B[:-1]),
                    )
                ),
                torch.zeros((p, (j + 1) * p), dtype=torch.double),
            )
        )
        + torch.vstack(
            (
                torch.zeros((p, (j + 1) * p), dtype=torch.double),
                torch.hstack(
                    (
                        torch.block_diag(*B[:-1]),
                        torch.zeros((j * p, p), dtype=torch.double),
                    )
                ),
            )
        )
    )
    V = torch.hstack(Q[:-1])

    return T, V


def randomized_eig(L, r, oversample=500, n_iter=0):
    n = L.shape[0]
    F = torch.randn(n, r + oversample)

    # Compute Y = A * F and apply optional power iterations
    Y = L @ F
    for _ in range(n_iter):
        Y = L @ Y

    # QR factorization
    Q, _ = torch.linalg.qr(Y, mode="reduced")

    # Form smaller matrix B
    B = Q.T @ L @ Q

    # Eigen decomposition of B
    vals, vecs = torch.linalg.eigh(B)

    # Sort by eigenvalues (descending)
    # idx = torch.argsort(vals)[::-1]
    # vals = vals[idx]
    # vecs = vecs[:, idx]

    # Take top r
    vals = vals[:r]
    vecs = vecs[:, :r]

    # Map back
    eigenvectors = Q @ vecs

    return vals, eigenvectors
