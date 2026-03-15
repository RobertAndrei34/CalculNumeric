import numpy as np


def backward_substitution(R, b, eps=1e-12):
    """
    Rezolvă sistemul superior triunghiular Rx = b.
    """
    n = R.shape[0]
    x = np.zeros(n, dtype=float)

    for i in range(n - 1, -1, -1):
        if abs(R[i, i]) < eps:
            raise ValueError(f"Matrice singulară sau aproape singulară: R[{i},{i}] ~ 0")
        s = np.dot(R[i, i + 1:], x[i + 1:])
        x[i] = (b[i] - s) / R[i, i]

    return x


def householder_qr(A, b=None, eps=1e-12):
    """
    Descompunere QR folosind algoritmul householder.
    """
    A = A.astype(float).copy()
    n = A.shape[0]

    Q_t = np.eye(n, dtype=float)
    b_t = None if b is None else b.astype(float).copy()

    for r in range(n - 1):
        # sigma = sum_{i=r}^{n-1} A[i, r]^2
        x = A[r:, r]
        sigma = np.dot(x, x)

        if sigma <= eps:
            # coloana deja "aproape zero" sub diagonala
            continue

        k = np.sqrt(sigma)
        if A[r, r] > 0:
            k = -k

        # construim vectorul u
        u = np.zeros(n, dtype=float)
        u[r] = A[r, r] - k
        u[r + 1:] = A[r + 1:, r]

        beta = np.dot(u, u)
        if beta <= eps:
            continue

        for j in range(r, n):
            gamma = np.dot(u[r:], A[r:, j]) / beta
            A[r:, j] = A[r:, j] - 2.0 * gamma * u[r:]

        if b_t is not None:
            gamma = np.dot(u[r:], b_t[r:]) / beta
            b_t[r:] = b_t[r:] - 2.0 * gamma * u[r:]


        for j in range(n):
            gamma = np.dot(u[r:], Q_t[r:, j]) / beta
            Q_t[r:, j] = Q_t[r:, j] - 2.0 * gamma * u[r:]

    R = A

    # verificare singularitate pe diagonala
    for i in range(n):
        if abs(R[i, i]) < eps:
            raise ValueError("Matrice singulara sau aproape singulara.")

    return Q_t, R, b_t


def solve_with_householder(A, b, eps=1e-12):
    """
    Rezolva Ax=b folosind QR householder implementat manual.
    """
    Q_t, R, b_t = householder_qr(A, b, eps)
    x = backward_substitution(R, b_t, eps)
    return x, Q_t, R, b_t


def solve_with_library_qr(A, b, eps=1e-12):
    """
    Rezolva Ax=b folosind QR din biblioteca.
    """
    Q, R = np.linalg.qr(A)
    y = Q.T @ b
    x = backward_substitution(R, y, eps)
    return x, Q, R


def inverse_with_householder(A, eps=1e-12):
    """
    Calculeaza A^{-1} folosind descompunerea QR householder.
    """
    n = A.shape[0]
    Q_t, R, _ = householder_qr(A, None, eps)

    A_inv = np.zeros((n, n), dtype=float)

    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0

        # b = Q^T e_j
        b = Q_t @ e_j
        x = backward_substitution(R, b, eps)
        A_inv[:, j] = x

    return A_inv


def generate_random_nonsingular_matrix(n, low=-10, high=10, eps=1e-12):
    """
    Genereaza o matrice patratica nesingulara
    """
    while True:
        A = np.random.uniform(low, high, (n, n))
        if abs(np.linalg.det(A)) > eps:
            return A


def solve_theme(n=4, eps=1e-12, A=None, s=None, random_init=False, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if random_init:
        A = generate_random_nonsingular_matrix(n, eps=eps)
        s = np.random.uniform(-10, 10, n)
    else:
        if A is None or s is None:
            raise ValueError("Pentru random_init=False trebuie date A și s.")
        A = np.array(A, dtype=float)
        s = np.array(s, dtype=float)
        n = A.shape[0]

    A_init = A.copy()
    s_init = s.copy()

    # 1. b = A * s
    b = A_init @ s_init

    # 2 + 3. Householder si cu biblioteca
    x_house, Q_t, R, b_house = solve_with_householder(A_init, b, eps)
    x_qr, Q_lib, R_lib = solve_with_library_qr(A_init, b, eps)

    # 4. erori
    err_house_res = np.linalg.norm(A_init @ x_house - b, 2)
    err_qr_res = np.linalg.norm(A_init @ x_qr - b, 2)

    err_house_rel = np.linalg.norm(x_house - s_init, 2) / np.linalg.norm(s_init, 2)
    err_qr_rel = np.linalg.norm(x_qr - s_init, 2) / np.linalg.norm(s_init, 2)

    diff_solutions = np.linalg.norm(x_qr - x_house, 2)

    # 5. inversa
    A_inv_house = inverse_with_householder(A_init, eps)
    A_inv_lib = np.linalg.inv(A_init)
    inv_diff = np.linalg.norm(A_inv_house - A_inv_lib, ord=1)

    # afisare rezultate
    print("===== DATE DE INTRARE =====")
    print("n =", n)
    print("eps =", eps)
    print("A =\n", A_init)
    print("s =", s_init)

    print("\n===== ex 3 =====")
    print("xQR =", x_qr)
    print("xHouseholder =", x_house)
    print("||xQR - xHouseholder||2 =", diff_solutions)

    print("\n===== ex 4 =====")
    print("||A_init * xHouseholder - b_init||2 =", err_house_res)
    print("||A_init * xQR - b_init||2 =", err_qr_res)
    print("||xHouseholder - s||2 / ||s||2 =", err_house_rel)
    print("||xQR - s||2 / ||s||2 =", err_qr_rel)

    print("\n===== ex 5 =====")
    print("A^{-1} Householder =\n", A_inv_house)
    print("A^{-1} bibliotecă =\n", A_inv_lib)
    print("||A^{-1}_Householder - A^{-1}_bibl||1 =", inv_diff)

    return {
        "A": A_init,
        "s": s_init,
        "b": b,
        "Q_t_house": Q_t,
        "R_house": R,
        "x_qr": x_qr,
        "x_house": x_house,
        "diff_solutions": diff_solutions,
        "err_house_res": err_house_res,
        "err_qr_res": err_qr_res,
        "err_house_rel": err_house_rel,
        "err_qr_rel": err_qr_rel,
        "A_inv_house": A_inv_house,
        "A_inv_lib": A_inv_lib,
        "inv_diff": inv_diff
    }


if __name__ == "__main__":
    # Exemplul din enunț
    A_example = np.array([
        [0, 0, 4],
        [1, 2, 3],
        [0, 1, 2]
    ], dtype=float)

    s_example = np.array([3, 2, 1], dtype=float)

    solve_theme(A=A_example, s=s_example, eps=1e-12, random_init=False)

    print("\n\n===== TEST CU DATE RANDOM =====")
    solve_theme(n=5, eps=1e-12, random_init=True, seed=42)