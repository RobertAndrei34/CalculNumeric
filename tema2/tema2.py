import numpy as np

try:
    from scipy.linalg import lu
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def generate_spd_matrix(n: int, seed: int | None = None) -> np.ndarray:
    """
    generare de matrice simetrica
        A = B * B^T + n * I
    """
    rng = np.random.default_rng(seed)
    B = rng.uniform(-5.0, 5.0, size=(n, n))
    A = B @ B.T
    A += n * np.eye(n)
    return A


def generate_vector_b(n: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-10.0, 10.0, size=n)


def print_matrix(name: str, M: np.ndarray, precision: int = 6) -> None:
    np.set_printoptions(precision=precision, suppress=True)
    print(f"{name} =")
    print(M)
    print()


def print_vector(name: str, v: np.ndarray, precision: int = 6) -> None:
    np.set_printoptions(precision=precision, suppress=True)
    print(f"{name} = {v}")
    print()


def ldlt_inplace(A: np.ndarray, eps: float) -> np.ndarray:
    """
    descompunere LDL^T
    """
    n = A.shape[0]
    d = np.zeros(n, dtype=float)

    for p in range(n):
        s = 0.0
        for k in range(p):
            s += d[k] * (A[p, k] ** 2)

        d[p] = A[p, p] - s

        if abs(d[p]) <= eps:
            raise ValueError(f"Nu se poate continua: d[{p}] = {d[p]}")

        for i in range(p + 1, n):
            s = 0.0
            for k in range(p):
                s += d[k] * A[i, k] * A[p, k]
            A[i, p] = (A[i, p] - s) / d[p]

    return d


def reconstruct_L(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    L = np.eye(n)
    for i in range(1, n):
        for j in range(i):
            L[i, j] = A[i, j]
    return L


def reconstruct_D(d: np.ndarray) -> np.ndarray:
    return np.diag(d)


def determinant_from_ldlt(d: np.ndarray) -> float:
    return float(np.prod(d))


def forward_substitution_unit_lower(A: np.ndarray, b: np.ndarray) -> np.ndarray:

    n = A.shape[0]
    z = np.zeros(n, dtype=float)

    for i in range(n):
        s = 0.0
        for j in range(i):
            s += A[i, j] * z[j]
        z[i] = b[i] - s

    return z


def solve_diagonal(d: np.ndarray, z: np.ndarray, eps: float) -> np.ndarray:

    n = len(d)
    y = np.zeros(n, dtype=float)

    for i in range(n):
        if abs(d[i]) <= eps:
            raise ValueError(f"Împărțire imposibilă: d[{i}] = {d[i]}")
        y[i] = z[i] / d[i]

    return y


def backward_substitution_unit_upper_from_lower(A: np.ndarray, y: np.ndarray) -> np.ndarray:

    n = A.shape[0]
    x = np.zeros(n, dtype=float)

    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += A[j, i] * x[j]
        x[i] = y[i] - s

    return x


def solve_with_ldlt(A: np.ndarray, d: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    z = forward_substitution_unit_lower(A, b)
    y = solve_diagonal(d, z, eps)
    x = backward_substitution_unit_upper_from_lower(A, y)
    return x


def manual_Ainit_times_x_from_compact_storage(A: np.ndarray, x: np.ndarray) -> np.ndarray:

    n = A.shape[0]
    y = np.zeros(n, dtype=float)

    for i in range(n):
        s = 0.0
        for j in range(n):
            if i <= j:
                aij = A[i, j]
            else:
                aij = A[j, i]
            s += aij * x[j]
        y[i] = s

    return y


def library_lu_and_solve(A: np.ndarray, b: np.ndarray):

    xlib = np.linalg.solve(A, b)

    if SCIPY_AVAILABLE:
        P, L, U = lu(A)
        return P, L, U, xlib

    return None, None, None, xlib


def verify_solution(A_compact: np.ndarray, x_chol: np.ndarray, b: np.ndarray, x_lib: np.ndarray):
    Ax = manual_Ainit_times_x_from_compact_storage(A_compact, x_chol)
    norm_residual = np.linalg.norm(Ax - b, ord=2)
    norm_diff = np.linalg.norm(x_chol - x_lib, ord=2)
    return float(norm_residual), float(norm_diff)


def verify_determinant(A_original: np.ndarray, d: np.ndarray):
    det_chol = determinant_from_ldlt(d)
    det_numpy = np.linalg.det(A_original)
    diff = abs(det_chol - det_numpy)

    print("\n================== VERIFICARE DETERMINANT ==================")
    print(f"det(A) din LDL^T = {det_chol}")
    print(f"det(A) din numpy = {det_numpy}")
    print(f"|det_LDLT - det_numpy| = {diff}")

    if diff < 1e-6:
        print("Determinantul este corect ✔")
    else:
        print("Determinantul NU coincide ✘")


def main():
    print("Tema 2 - Descompunere LDL^T, LU, determinant, rezolvare sistem")
    print()

    n = int(input("n = ").strip())
    eps = float(input("eps = ").strip())

    A = generate_spd_matrix(n, seed=42)
    b = generate_vector_b(n, seed=123)

    A_original = A.copy()

    P, L_lu, U_lu, x_lib = library_lu_and_solve(A_original, b)

    print("\n================== DATE INIȚIALE ==================")
    if n <= 10:
        print_matrix("A", A_original)
        print_vector("b", b)
    else:
        print(f"Matrice A generată: {n} x {n}")
        print(f"Vector b generat: dimensiune {n}")

    print("\n================== DESCOMPUNERE LU ==================")
    if SCIPY_AVAILABLE:
        if n <= 10:
            print_matrix("L (LU)", L_lu)
            print_matrix("U (LU)", U_lu)
        else:
            print("Descompunerea LU a fost calculată cu succes.")
    else:
        print("SciPy nu este instalat, LU nu poate fi afișată explicit.")
        print("Soluția xlib a fost calculată cu numpy.linalg.solve.")

    if n <= 10:
        print_vector("xlib", x_lib)

    d = ldlt_inplace(A, eps)

    print("\n================== DESCOMPUNERE LDL^T ==================")
    if n <= 10:
        L = reconstruct_L(A)
        D = reconstruct_D(d)
        print_matrix("L", L)
        print_matrix("D", D)
        print_vector("d", d)
    else:
        print("Descompunerea LDL^T a fost calculată cu succes.")

    det_A = determinant_from_ldlt(d)

    print("\n================== DETERMINANT ==================")
    print(f"det(A) din LDL^T = {det_A}")

    verify_determinant(A_original, d)

    x_chol = solve_with_ldlt(A, d, b, eps)

    print("\n================== SOLUȚIE CU LDL^T ==================")
    if n <= 10:
        print_vector("xChol", x_chol)
    else:
        print("xChol a fost calculat cu succes.")

    norm_residual, norm_diff = verify_solution(A, x_chol, b, x_lib)

    print("\n================== VERIFICARE SOLUȚIE ==================")
    print(f"||Ainit * xChol - b||_2 = {norm_residual:e}")
    print(f"||xChol - xlib||_2     = {norm_diff:e}")

    if norm_residual < 1e-8:
        print("Reziduul este suficient de mic ✔")
    else:
        print("Reziduul NU este suficient de mic ✘")

    if norm_diff < 1e-9:
        print("Soluția coincide cu cea de bibliotecă ✔")
    else:
        print("Soluția diferă de cea de bibliotecă ✘")


if __name__ == "__main__":
    main()