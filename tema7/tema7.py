from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable


KMAX = 1000
DELTA_MAX = 10**8


@dataclass
class MethodResult:
    method: str
    x0: float
    root: float | None
    steps: int
    converged: bool
    reason: str


def horner(coeffs: list[float], x: float) -> float:
    """
    calculare P(x) prin schema lui Horner
    """
    if not coeffs:
        raise ValueError("Lista de coeficienti nu poate fi vida.")

    b = coeffs[0]
    for a in coeffs[1:]:
        b = b * x + a
    return b


def derivative_coeffs(coeffs: list[float]) -> list[float]:
    """
    returnare coeficientii derivatei polinomului
    """
    n = len(coeffs) - 1
    if n <= 0:
        return [0.0]

    return [coeffs[i] * (n - i) for i in range(n)]


def root_bound(coeffs: list[float]) -> float:

    if not coeffs:
        raise ValueError("Lista de coeficienti nu poate fi vida.")
    if abs(coeffs[0]) == 0:
        raise ValueError("Coeficientul dominant a0 trebuie sa fie nenul.")

    if len(coeffs) == 1:
        return 0.0

    a0 = coeffs[0]
    A = max(abs(a) for a in coeffs[1:])
    return (abs(a0) + A) / abs(a0)


def newton_step(coeffs: list[float], d1: list[float], d2: list[float], x: float, eps: float) -> tuple[float | None, str]:

    px = horner(coeffs, x)
    p1 = horner(d1, x)

    if abs(p1) <= eps:
        return None, "|P'(x)| <= eps"

    return px / p1, "ok"


def olver_step(coeffs: list[float], d1: list[float], d2: list[float], x: float, eps: float) -> tuple[float | None, str]:

    px = horner(coeffs, x)
    p1 = horner(d1, x)
    p2 = horner(d2, x)

    if abs(p1) <= eps:
        return None, "|P'(x)| <= eps"

    correction = (px * px * p2) / (p1 * p1 * p1)
    return px / p1 + 0.5 * correction, "ok"


def approximate_root(
    coeffs: list[float],
    x0: float,
    eps: float,
    method: str,
    kmax: int = KMAX,
    delta_max: float = DELTA_MAX,
) -> MethodResult:
    """
    se plica Newton sau Olver plecand din x0 cu criteriul de oprire
        |delta_x| < eps
        k > kmax sau |delta_x| > 10^8 sau derivata prea mica
    """
    d1 = derivative_coeffs(coeffs)
    d2 = derivative_coeffs(d1)

    if method.lower() == "newton":
        step_function = newton_step
        method_name = "Newton"
    elif method.lower() == "olver":
        step_function = olver_step
        method_name = "Olver"
    else:
        raise ValueError("Metoda trebuie sa fie 'newton' sau 'olver'.")

    x = x0
    delta = math.inf

    for k in range(kmax + 1):
        delta, reason = step_function(coeffs, d1, d2, x, eps)

        if delta is None:
            return MethodResult(method_name, x0, None, k, False, reason)

        if abs(delta) > delta_max:
            return MethodResult(method_name, x0, None, k, False, "|delta_x| > 10^8")

        x = x - delta

        if abs(delta) < eps:

            if abs(horner(coeffs, x)) <= max(1.0, sum(abs(c) for c in coeffs)) * eps * 100:
                return MethodResult(method_name, x0, x, k + 1, True, "convergent")
            return MethodResult(method_name, x0, x, k + 1, False, "|P(x)| nu este suficient de mic")

    return MethodResult(method_name, x0, None, kmax, False, "kmax depasit")


def is_distinct(value: float, roots: list[float], eps: float) -> bool:

    return all(abs(value - r) > eps for r in roots)


def distinct_roots_from_results(results: list[MethodResult], eps: float) -> list[float]:
    """
    Extrage radacinile distincte dintr-o lista de rezultate.
    """
    roots: list[float] = []

    for result in results:
        if result.converged and result.root is not None:
            if is_distinct(result.root, roots, eps):
                roots.append(result.root)

    roots.sort()
    return roots


def generate_start_points(R: float, count_grid: int = 250, count_random: int = 250) -> list[float]:
    """
      puncte echidistante
      puncte aleatoare
    """
    points: list[float] = []

    if R == 0:
        return [0.0]

    if count_grid < 2:
        count_grid = 2

    step = 2 * R / (count_grid - 1)
    for i in range(count_grid):
        points.append(-R + i * step)

    random.seed(7)
    for _ in range(count_random):
        points.append(random.uniform(-R, R))

    return points


def solve_polynomial(
    coeffs: list[float],
    eps: float = 1e-10,
    output_file: str = "radacini.txt",
    count_grid: int = 250,
    count_random: int = 250,
) -> tuple[list[MethodResult], list[MethodResult], list[float]]:
    """
    se ruleaza Newton si Olver din mai multe puncte de start
    """
    if eps <= 0:
        raise ValueError("Precizia eps trebuie sa fie pozitiva.")
    if len(coeffs) < 2:
        raise ValueError("Polinomul trebuie sa aiba grad cel putin 1.")
    if abs(coeffs[0]) == 0:
        raise ValueError("Coeficientul dominant a0 trebuie sa fie nenul.")

    R = root_bound(coeffs)
    start_points = generate_start_points(R, count_grid=count_grid, count_random=count_random)

    newton_results: list[MethodResult] = []
    olver_results: list[MethodResult] = []

    for x0 in start_points:
        newton_results.append(approximate_root(coeffs, x0, eps, "newton"))
        olver_results.append(approximate_root(coeffs, x0, eps, "olver"))

    all_results = newton_results + olver_results
    roots = distinct_roots_from_results(all_results, eps)

    print("=" * 80)
    print("Polinom:", coeffs)
    print(f"eps = {eps:g}")
    print(f"R = {R:.16g}")
    print(f"Toate radacinile reale se afla in intervalul [{-R:.16g}, {R:.16g}]")
    print("=" * 80)

    print("\nRadacini reale distincte gasite:")
    if roots:
        for i, root in enumerate(roots, 1):
            print(f"  r{i} = {root:.16g}, P(r{i}) = {horner(coeffs, root):.3e}")
    else:
        print("  Nu s-au gasit radacini reale cu punctele de start folosite.")

    print("\nComparatie Newton vs Olver pentru fiecare radacina gasita:")
    print("-" * 80)
    print(f"{'radacina':>22} | {'pasi Newton':>12} | {'pasi Olver':>10}")
    print("-" * 80)

    for root in roots:
        n_steps = [
            result.steps
            for result in newton_results
            if result.converged and result.root is not None and abs(result.root - root) <= eps
        ]
        o_steps = [
            result.steps
            for result in olver_results
            if result.converged and result.root is not None and abs(result.root - root) <= eps
        ]

        best_newton = min(n_steps) if n_steps else None
        best_olver = min(o_steps) if o_steps else None

        print(
            f"{root:22.15g} | "
            f"{str(best_newton):>12} | "
            f"{str(best_olver):>10}"
        )

    print("-" * 80)

    with open(output_file, "w", encoding="utf-8") as fout:
        for root in roots:
            fout.write(f"{root:.16g}\n")

    print(f"\nRadacinile distincte au fost salvate in fisierul: {output_file}")

    return newton_results, olver_results, roots


def read_polynomial_from_keyboard() -> tuple[list[float], float]:
    """
    citire
      grad n
      coeficientii a0 a1 ... an
      precizia eps
    """
    n = int(input("Gradul polinomului n = ").strip())
    coeffs = []

    print("Introdu coeficientii a0, a1, ..., an:")
    while len(coeffs) < n + 1:
        line = input().strip()
        if not line:
            continue
        coeffs.extend(float(x) for x in line.split())

    coeffs = coeffs[: n + 1]
    eps = float(input("Precizia eps = ").strip())

    return coeffs, eps


def demo_examples() -> None:
    """
   exemplele din enunt
    """
    examples = [
        [1.0, -6.0, 11.0, -6.0],
        [42.0, -55.0, -42.0, 49.0, -6.0],
        [8.0, -38.0, 49.0, -22.0, 3.0],
        [1.0, -6.0, 13.0, -12.0, 4.0],
    ]

    eps = 1e-10

    for index, coeffs in enumerate(examples, 1):
        output_file = f"radacini_exemplul_{index}.txt"
        solve_polynomial(
            coeffs,
            eps=eps,
            output_file=output_file,
            count_grid=300,
            count_random=300,
        )
        print("\n\n")


if __name__ == "__main__":
    print("Alege modul de rulare:")
    print("1 - introducere de la tastatura")
    print("2 - exemplele din enunt")
    option = input("Optiune = ").strip()

    if option == "1":
        coeffs, eps = read_polynomial_from_keyboard()
        solve_polynomial(coeffs, eps=eps, output_file="radacini.txt")
    else:
        demo_examples()
