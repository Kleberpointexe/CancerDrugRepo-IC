def nmf_two_matrices(A, B, k, tol=1e-4, eps=1e-9, seed=0, max_iter=10000, logger=None):
    import numpy as np

    np.random.seed(seed)
    m_A, n_A = A.shape
    m_B, n_B = B.shape

    Wd = np.random.rand(m_A, k) * 1e-3
    Wc = np.random.rand(m_B, k) * 1e-3
    H = np.random.rand(k, n_A) * 1e-3

    error_history = []
    it = 0
    diff = np.inf

    while diff > tol and it < max_iter:

        error_A = np.linalg.norm(A - Wd @ H, 'fro')**2
        error_B = np.linalg.norm(B - Wc @ H, 'fro')**2
        current_error = error_A + error_B
        error_history.append(current_error)

        if it > 0:
            prev_error = error_history[-2]
            diff = abs(prev_error - current_error) / (prev_error + eps)

        HHt = H @ H.T

        Wd *= (A @ H.T) / (Wd @ HHt + eps)
        Wc *= (B @ H.T) / (Wc @ HHt + eps)

        num = (Wd.T @ A) + (Wc.T @ B)
        den = (Wd.T @ Wd + Wc.T @ Wc) @ H + eps
        H *= num / den

        it += 1

    if logger:
        logger.info(
            f"NMF k={k} convergiu em {it} iterações | erro={current_error:.6f} | diff={diff:.6e}"
        )

    return Wd, Wc, H, error_history, it


def test_multiple_k(A, B, k_values, tol=1e-4, max_iter=10000, logger=None):
    results = {}

    for k in k_values:
        if logger:
            logger.info(f"Treinando k={k}")

        results[k] = nmf_two_matrices(A, B, k, tol=tol, max_iter=max_iter, logger=logger)

    return results