
import numpy as np

def is_symmetric(A, tol=1e-12):
    return np.allclose(A, A.T, atol=tol, rtol=0.0)

def offdiag_max_abs(A):
    """
    Return (p, q, value) where |A[p,q]| is maximal for p>q (strict lower triangle).
    """
    n = A.shape[0]
    p, q = 1, 0
    max_val = 0.0
    for i in range(1, n):
        for j in range(i):
            v = abs(A[i, j])
            if v > max_val:
                max_val = v
                p, q = i, j
    return p, q, A[p, q]

def jacobi_eigen(A, eps=1e-12, kmax=10_000):
    """
    Jacobi method for eigenvalues/eigenvectors of a real symmetric matrix.
    Returns:
        lambdas  - approximate eigenvalues (diagonal of final matrix)
        U        - approximate eigenvectors as columns
        A_final  - approximately diagonal matrix U^T A_init U
        iterations
    """
    A = np.array(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Jacobi method requires a square matrix.")
    if not is_symmetric(A):
        raise ValueError("Jacobi method requires a symmetric matrix.")

    n = A.shape[0]
    A_init = A.copy()
    U = np.eye(n)

    p, q, apq = offdiag_max_abs(A)
    k = 0

    while abs(apq) > eps and k < kmax:
        app = A[p, p]
        aqq = A[q, q]

        # Formula
        # alpha = (app - aqq) / (2 * apq)
        alpha = (app - aqq) / (2.0 * apq)

        # Stable choice for t
        if alpha >= 0:
            t = -alpha + np.sqrt(alpha * alpha + 1.0)
        else:
            t = -alpha - np.sqrt(alpha * alpha + 1.0)

        c = 1.0 / np.sqrt(1.0 + t * t)
        s = t / np.sqrt(1.0 + t * t)

        # Update A in-place using formulas from the statement
        for j in range(n):
            if j != p and j != q:
                apj_old = A[p, j]
                aqj_old = A[q, j]

                A[p, j] = c * apj_old + s * aqj_old
                A[j, p] = A[p, j]

                A[q, j] = -s * apj_old + c * aqj_old
                A[j, q] = A[q, j]

        A[p, p] = app + t * apq
        A[q, q] = aqq - t * apq
        A[p, q] = 0.0
        A[q, p] = 0.0

        # Update eigenvector matrix U columns p and q
        u_p_old = U[:, p].copy()
        u_q_old = U[:, q].copy()
        U[:, p] = c * u_p_old + s * u_q_old
        U[:, q] = -s * u_p_old + c * u_q_old

        p, q, apq = offdiag_max_abs(A)
        k += 1

    lambdas = np.diag(A).copy()
    return lambdas, U, A, k, A_init

def verify_jacobi(A_init, lambdas, U):
    Lambda = np.diag(lambdas)
    residual = A_init @ U - U @ Lambda
    return np.linalg.norm(residual)

def cholesky_iteration(A, eps=1e-12, kmax=1000):
    """
    A^(0) = A = L0 L0^T
    A^(k+1) = Lk^T Lk
    Stop when ||A^(k) - A^(k-1)|| < eps or k > kmax.
    Works for symmetric positive definite matrices.
    """
    A = np.array(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Cholesky iteration requires a square matrix.")
    if not is_symmetric(A):
        raise ValueError("Cholesky iteration requires a symmetric matrix.")

    A_curr = A.copy()
    for k in range(1, kmax + 1):
        L = np.linalg.cholesky(A_curr)
        A_next = L.T @ L
        diff = np.linalg.norm(A_next - A_curr)
        if diff < eps:
            return A_next, k, diff
        A_curr = A_next

    return A_curr, kmax, np.linalg.norm(A_curr - (np.linalg.cholesky(A_curr).T @ np.linalg.cholesky(A_curr)))

def svd_analysis(A, eps=1e-12):
    """
    For p > n:
      - singular values
      - rank
      - condition number
      - Moore-Penrose pseudoinverse AI = V * SI * U^T
      - least-squares pseudoinverse AJ = (A^T A)^(-1) A^T (when possible)
      - ||AI - AJ||_1
    """
    A = np.array(A, dtype=float)
    p, n = A.shape
    if p <= n:
        raise ValueError("SVD analysis from the assignment is intended for p > n.")

    U, s, VT = np.linalg.svd(A, full_matrices=True)

    rank = int(np.sum(s > eps))

    positive_s = s[s > eps]
    cond_manual = np.inf if len(positive_s) == 0 else positive_s.max() / positive_s.min()
    cond_lib = np.linalg.cond(A) if rank > 0 else np.inf

    # Moore-Penrose pseudoinverse via SVD
    SI = np.zeros((n, p))
    for i in range(rank):
        SI[i, i] = 1.0 / s[i]
    AI = VT.T @ SI @ U.T

    # Least-squares pseudoinverse
    ATA = A.T @ A
    AJ = None
    diff_norm_1 = None
    try:
        AJ = np.linalg.inv(ATA) @ A.T
        diff_norm_1 = np.linalg.norm(AI - AJ, 1)
    except np.linalg.LinAlgError:
        AJ = None
        diff_norm_1 = None

    return {
        "singular_values": s,
        "rank_manual": rank,
        "rank_lib": np.linalg.matrix_rank(A),
        "condition_manual": cond_manual,
        "condition_lib": cond_lib,
        "AI_moore_penrose": AI,
        "AJ_least_squares": AJ,
        "norm1_AI_minus_AJ": diff_norm_1,
    }

def pretty_matrix(M, precision=6):
    return np.array2string(np.array(M, dtype=float), precision=precision, suppress_small=True)

def solve_assignment(A, eps=1e-10, kmax=1000):
    """
    Solves all requested items depending on matrix shape.
    """
    A = np.array(A, dtype=float)
    p, n = A.shape

    print("=" * 80)
    print("Matricea A:")
    print(pretty_matrix(A))
    print(f"Dimensiune: p={p}, n={n}")
    print(f"Epsilon = {eps}, kmax = {kmax}")
    print("=" * 80)

    if p == n:
        print("\nCAZUL p = n")
        if not is_symmetric(A):
            print("Matricea NU este simetrică, deci metoda Jacobi și iterația Cholesky")
            print("din cerință nu se aplică direct.")
            return

        # 1) Jacobi
        print("\n1) Metoda Jacobi pentru valori/vectori proprii")
        lambdas, U, A_final, iterations, A_init = jacobi_eigen(A, eps=eps, kmax=kmax)
        residual_norm = verify_jacobi(A_init, lambdas, U)

        print(f"Număr iterații Jacobi: {iterations}")
        print("Matricea finală (aprox. diagonală):")
        print(pretty_matrix(A_final))
        print("Valorile proprii aproximative (diagonala):")
        print(pretty_matrix(lambdas))
        print("Vectorii proprii aproximativi (coloanele lui U):")
        print(pretty_matrix(U))
        print("Norma ||A_init * U - U * Lambda||:")
        print(residual_norm)

        # 2) Cholesky iteration
        print("\n2) Șirul de matrice bazat pe factorizarea Cholesky")
        try:
            A_last, steps, final_diff = cholesky_iteration(A, eps=eps, kmax=kmax)
            print(f"Număr pași: {steps}")
            print("Ultima matrice calculată:")
            print(pretty_matrix(A_last))
            print(f"||A^(k) - A^(k-1)|| = {final_diff}")

            print("\nObservație:")
            print("Ultima matrice tinde către o matrice diagonală (sau aproape diagonală),")
            print("iar pe diagonală apar valorile proprii ale matricei inițiale.")
        except np.linalg.LinAlgError as e:
            print("Factorizarea Cholesky nu poate fi aplicată acestei matrice:")
            print(str(e))
            print("Asta se întâmplă când matricea nu este pozitiv definită.")

    elif p > n:
        print("\nCAZUL p > n")
        print("\n3) Analiza SVD")
        result = svd_analysis(A, eps=eps)

        print("Valorile singulare:")
        print(pretty_matrix(result["singular_values"]))
        print(f"Rang (manual, din valori singulare > eps): {result['rank_manual']}")
        print(f"Rang (funcție bibliotecă): {result['rank_lib']}")
        print(f"Număr de condiționare (manual): {result['condition_manual']}")
        print(f"Număr de condiționare (funcție bibliotecă): {result['condition_lib']}")

        print("\nPseudoinversa Moore-Penrose AI:")
        print(pretty_matrix(result["AI_moore_penrose"]))

        if result["AJ_least_squares"] is not None:
            print("\nPseudo-inversa în sensul celor mai mici pătrate AJ = (A^T A)^(-1) A^T:")
            print(pretty_matrix(result["AJ_least_squares"]))
            print("\nNorma ||AI - AJ||_1:")
            print(result["norm1_AI_minus_AJ"])
        else:
            print("\nAJ nu poate fi calculată cu formula (A^T A)^(-1) A^T deoarece A^T A este singulară.")
    else:
        print("\nCazul p < n nu este cerut în această temă.")

def main():
    eps = 1e-10
    kmax = 5000

    # EXEMPLE din PDF / exemple utile

    # Ex. 1: matrice simetrică 3x3
    A1 = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [1, 1, 1]
    ], dtype=float)
    solve_assignment(A1, eps=eps, kmax=kmax)

    print("\n\n")

    # Ex. 2: matrice simetrică 4x4
    A2 = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ], dtype=float)
    solve_assignment(A2, eps=eps, kmax=kmax)

    print("\n\n")

    # Ex. 3: caz p > n
    A3 = np.array([
        [1, 2, 0],
        [0, 1, 1],
        [1, 1, 1],
        [2, 0, 1],
        [3, 1, 2]
    ], dtype=float)
    solve_assignment(A3, eps=eps, kmax=kmax)

if __name__ == "__main__":
    main()
