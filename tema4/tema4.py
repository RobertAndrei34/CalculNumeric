from __future__ import annotations

from pathlib import Path
import sys


# ============================================================================
# Citire date
# ============================================================================

def read_vector(path: str | Path) -> list[float]:
    """Citește un vector numeric din fișier text."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [float(tok.replace(",", ".")) for tok in text.split()]


# ============================================================================
# Utilitare pentru reprezentarea rară
# ============================================================================

def infer_diagonal_order(n: int, diag_len: int) -> int:
    """
    Pentru o diagonală secundară superioară/inferioară de ordin k,
    lungimea vectorului memorat este n - k.
    Deci k = n - len(diag).
    """
    k = n - diag_len
    if not (1 <= k <= n - 1):
        raise ValueError(
            f"Lungime invalidă pentru o diagonală secundară: len={diag_len}, n={n}."
        )
    return k


def validate_main_diagonal(d0: list[float], eps: float) -> bool:
    """Verifică dacă toate elementele de pe diagonala principală sunt nenule."""
    return all(abs(v) > eps for v in d0)


# ============================================================================
# Operații cu matricea rară simetrică
# ============================================================================

def multiply_sparse_symmetric(
    d0: list[float],
    d1: list[float],
    d2: list[float],
    x: list[float],
) -> list[float]:
    """
    Calculează y = A*x folosind doar vectorii d0, d1, d2.
    Matricea este simetrică, cu diagonalele secundare de ordin p și q.
    """
    n = len(d0)
    if len(x) != n:
        raise ValueError("Vectorii d0 și x trebuie să aibă aceeași dimensiune.")

    p = infer_diagonal_order(n, len(d1))
    q = infer_diagonal_order(n, len(d2))

    y = [0.0] * n

    # diagonala principală
    for i, aii in enumerate(d0):
        y[i] += aii * x[i]

    # diagonalele de ordin p
    for i, a in enumerate(d1):
        y[i] += a * x[i + p]
        y[i + p] += a * x[i]

    # diagonalele de ordin q
    for i, a in enumerate(d2):
        y[i] += a * x[i + q]
        y[i + q] += a * x[i]

    return y


# ============================================================================
# Metoda Gauss-Seidel pentru matrice rară simetrică
# ============================================================================

def gauss_seidel_sparse(
    d0: list[float],
    d1: list[float],
    d2: list[float],
    b: list[float],
    eps: float = 1e-8,
    kmax: int = 10_000,
    divergence_limit: float = 1e10,
) -> dict:
    n = len(d0)
    if len(b) != n:
        raise ValueError("Vectorii d0 și b trebuie să aibă aceeași dimensiune.")

    p = infer_diagonal_order(n, len(d1))
    q = infer_diagonal_order(n, len(d2))

    if not validate_main_diagonal(d0, eps):
        return {
            "success": False,
            "reason": "Există element(e) nule pe diagonala principală.",
            "n": n,
            "p": p,
            "q": q,
        }

    # x este actualizat in-place -> variantă Gauss-Seidel cu un singur vector
    x = [0.0] * n
    iterations = 0
    delta = float("inf")

    while iterations < kmax:
        delta = 0.0

        for i in range(n):
            s = b[i]

            # contribuțiile de pe diagonala de ordin p
            if i - p >= 0:
                s -= d1[i - p] * x[i - p]   # valoare deja actualizată la iterația curentă
            if i + p < n:
                s -= d1[i] * x[i + p]       # valoare încă din iterația precedentă

            # contribuțiile de pe diagonala de ordin q
            if i - q >= 0:
                s -= d2[i - q] * x[i - q]
            if i + q < n:
                s -= d2[i] * x[i + q]

            new_xi = s / d0[i]
            diff = abs(new_xi - x[i])
            if diff > delta:
                delta = diff
            x[i] = new_xi

        iterations += 1

        if delta < eps:
            y = multiply_sparse_symmetric(d0, d1, d2, x)
            norm_inf = max(abs(y[i] - b[i]) for i in range(n)) if n else 0.0
            return {
                "success": True,
                "reason": None,
                "n": n,
                "p": p,
                "q": q,
                "x": x,
                "y": y,
                "iterations": iterations,
                "delta": delta,
                "norm_inf": norm_inf,
            }

        if delta > divergence_limit:
            return {
                "success": False,
                "reason": "Divergență numerică (delta a depășit 1e10).",
                "n": n,
                "p": p,
                "q": q,
                "iterations": iterations,
                "delta": delta,
                "x": None,
            }

    return {
        "success": False,
        "reason": "Limita de iterații a fost atinsă.",
        "n": n,
        "p": p,
        "q": q,
        "iterations": iterations,
        "delta": delta,
        "x": None,
    }


# ============================================================================
# Afișare rezultate
# ============================================================================

def print_report(
    name: str,
    result: dict,
    d0: list[float],
    d1: list[float],
    d2: list[float],
    b: list[float],
    eps: float,
) -> None:
    n = result["n"]
    p = result["p"]
    q = result["q"]

    x_upper = n - p - 1
    y_lower = n - q - 1

    print("=" * 80)
    print(f"Sistem: {name}")
    print(f"1) Dimensiunea sistemului n = {n}")
    print(f"2) Diagonala din d1 are ordinul p = {p}")
    print(f"   Relația pentru diagonala superioară: x = n - p - 1 = {x_upper}")
    print(f"   Diagonala din d2 are ordinul q = {q}")
    print(f"   Relația pentru diagonala inferioară: y = n - q - 1 = {y_lower}")
    print(f"3) Toate elementele din d0 sunt nenule? {'DA' if validate_main_diagonal(d0, eps) else 'NU'}")

    if result["success"]:
        print(f"4) Soluția a fost aproximată cu Gauss-Seidel în {result['iterations']} iterații.")
        print("   x_GS =")
        preview = min(20, len(result["x"]))
        for i in range(preview):
            print(f"   x[{i}] = {result['x'][i]:.12g}")
        if len(result["x"]) > preview:
            print(f"   ... ({len(result['x']) - preview} valori în plus)")

        print("5) y = A * x_GS =")
        preview = min(20, len(result["y"]))
        for i in range(preview):
            print(f"   y[{i}] = {result['y'][i]:.12g}")
        if len(result["y"]) > preview:
            print(f"   ... ({len(result['y']) - preview} valori în plus)")

        print(f"6) ||A*x_GS - b||_inf = {result['norm_inf']:.12g}")
    else:
        print(f"4) Sistemul NU a putut fi rezolvat: {result['reason']}")
        print("5) y = A*x_GS nu se poate calcula deoarece nu avem x_GS.")
        print("6) Norma nu se poate calcula deoarece nu avem x_GS.")

    print("7) În toate calculele s-au folosit doar d0, d1, d2 și b.")
    print()


# ============================================================================
# Rezolvare pentru un sistem / toate sistemele
# ============================================================================

def solve_one_system(
    d0_path: str | Path,
    d1_path: str | Path,
    d2_path: str | Path,
    b_path: str | Path,
    eps: float = 1e-8,
    kmax: int = 10_000,
) -> dict:
    d0 = read_vector(d0_path)
    d1 = read_vector(d1_path)
    d2 = read_vector(d2_path)
    b = read_vector(b_path)

    result = gauss_seidel_sparse(d0, d1, d2, b, eps=eps, kmax=kmax)
    print_report(
        name=f"{Path(d0_path).name}, {Path(d1_path).name}, {Path(d2_path).name}, {Path(b_path).name}",
        result=result,
        d0=d0,
        d1=d1,
        d2=d2,
        b=b,
        eps=eps,
    )
    return result


def solve_all_from_folder(folder: str | Path, eps: float = 1e-8, count: int = 5) -> list[dict]:
    folder = Path(folder)
    results = []

    def pick_file(prefix: str, idx: int) -> Path:
        candidates = [
            folder / f"{prefix}_{idx}.txt",
            folder / f"{prefix} {idx}.txt",
            folder / f"{prefix}{idx}.txt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Nu am găsit fișier pentru {prefix}, index {idx}. "
            f"Am încercat: {', '.join(str(c) for c in candidates)}"
        )

    for i in range(1, count + 1):
        print(f"sistemul {i}...")
        try:
            d0_path = pick_file("d0", i)
            d1_path = pick_file("d1", i)
            d2_path = pick_file("d2", i)
            b_path = pick_file("b", i)
            result = solve_one_system(d0_path, d1_path, d2_path, b_path, eps=eps)
        except FileNotFoundError as exc:
            print("=" * 80)
            print(f"Sistem: index {i}")
            print(f"Eroare: {exc}")
            print()
            result = {"success": False, "reason": str(exc), "n": None, "p": None, "q": None}
        except Exception as exc:
            print("=" * 80)
            print(f"Sistem: index {i}")
            print(f"Eroare neașteptată: {exc}")
            print()
            result = {"success": False, "reason": str(exc), "n": None, "p": None, "q": None}

        results.append(result)

    return results


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path(".")
    eps = float(sys.argv[2]) if len(sys.argv) >= 3 else 1e-8
    solve_all_from_folder(folder, eps=eps, count=5)
