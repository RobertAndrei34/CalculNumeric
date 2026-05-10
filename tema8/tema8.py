from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Dict, List, Tuple
import math
import numpy as np

Array = np.ndarray


# -----------------------------
# utilitare numerice
# -----------------------------

def sigmoid(z: float) -> float:
    """Sigmoid stabil numeric."""
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def softplus(z: float) -> float:
    """log(1 + exp(z)), stabil numeric."""
    if z > 0:
        return z + math.log1p(math.exp(-z))
    return math.log1p(math.exp(z))


def norm2(x: Array) -> float:
    return float(np.linalg.norm(x, ord=2))


def finite_difference_gradient(
    f: Callable[[Array], float],
    x: Array,
    h: float = 1e-5,
) -> Array:

    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x, dtype=float)

    for i in range(len(x)):
        e = np.zeros_like(x, dtype=float)
        e[i] = 1.0
        f1 = f(x + 2.0 * h * e)
        f2 = f(x + h * e)
        f3 = f(x - h * e)
        f4 = f(x - 2.0 * h * e)
        g[i] = (-f1 + 8.0 * f2 - 8.0 * f3 + f4) / (12.0 * h)

    return g



@dataclass
class Objective:
    name: str
    f: Callable[[Array], float]
    grad: Callable[[Array], Array]
    x0: Array
    known_minimum: Optional[Array] = None
    note: str = ""


def logistic_loss(x: Array) -> float:
    w0, w1 = float(x[0]), float(x[1])
    return softplus(w0 - w1) + softplus(-(w0 + w1))


def grad_logistic_loss(x: Array) -> Array:
    w0, w1 = float(x[0]), float(x[1])
    s1 = sigmoid(w0 - w1)
    s2 = sigmoid(w0 + w1)
    return np.array([s1 + s2 - 1.0, s2 - s1 - 1.0], dtype=float)


def f2(x: Array) -> float:
    x1, x2 = x
    return x1**2 + x2**2 - 2.0 * x1 - 4.0 * x2 - 1.0


def grad_f2(x: Array) -> Array:
    x1, x2 = x
    return np.array([2.0 * x1 - 2.0, 2.0 * x2 - 4.0], dtype=float)


def f3(x: Array) -> float:
    x1, x2 = x
    return 3.0 * x1**2 - 12.0 * x1 + 2.0 * x2**2 + 16.0 * x2 - 10.0


def grad_f3(x: Array) -> Array:
    x1, x2 = x
    return np.array([6.0 * x1 - 12.0, 4.0 * x2 + 16.0], dtype=float)


def f4(x: Array) -> float:
    x1, x2 = x
    return x1**2 - 4.0 * x1 * x2 + 4.5 * x2**2 - 4.0 * x2 + 3.0


def grad_f4(x: Array) -> Array:
    x1, x2 = x
    return np.array([2.0 * x1 - 4.0 * x2, -4.0 * x1 + 9.0 * x2 - 4.0], dtype=float)


def f5(x: Array) -> float:
    x1, x2 = x
    return x1**2 * x2 - 2.0 * x1 * x2**2 + 3.0 * x1 * x2 + 4.0


def grad_f5(x: Array) -> Array:
    x1, x2 = x
    return np.array([
        2.0 * x1 * x2 - 2.0 * x2**2 + 3.0 * x2,
        x1**2 - 4.0 * x1 * x2 + 3.0 * x1,
    ], dtype=float)


OBJECTIVES: List[Objective] = [
    Objective(
        name="l(w0,w1) = -ln(1-sigma(w0-w1)) - ln(sigma(w0+w1))",
        f=logistic_loss,
        grad=grad_logistic_loss,
        x0=np.array([0.0, 0.0]),
        known_minimum=None,
        note="Nu are minim finit: infimumul este 0, atins doar la limita w0-w1 -> -inf si w0+w1 -> +inf.",
    ),
    Objective(
        name="F1(x1,x2) = x1^2 + x2^2 - 2x1 - 4x2 - 1",
        f=f2,
        grad=grad_f2,
        x0=np.array([5.0, -3.0]),
        known_minimum=np.array([1.0, 2.0]),
    ),
    Objective(
        name="F2(x1,x2) = 3x1^2 - 12x1 + 2x2^2 + 16x2 - 10",
        f=f3,
        grad=grad_f3,
        x0=np.array([-3.0, 3.0]),
        known_minimum=np.array([2.0, -4.0]),
    ),
    Objective(
        name="F3(x1,x2) = x1^2 - 4x1x2 + 4.5x2^2 - 4x2 + 3",
        f=f4,
        grad=grad_f4,
        x0=np.array([2.0, 8.0]),
        known_minimum=np.array([8.0, 4.0]),
    ),
    Objective(
        name="F4(x1,x2) = x1^2*x2 - 2x1*x2^2 + 3x1*x2 + 4",
        f=f5,
        grad=grad_f5,
        x0=np.array([-0.6, 0.8]),  # aproape de minimul local (-1, 0.5)
        known_minimum=np.array([-1.0, 0.5]),
        note="Functia este neconvexa; punctul (-1, 0.5) este minim local, nu neaparat global.",
    ),
]

def constant_learning_rate(eta: float) -> Callable[[Callable[[Array], float], Array, Array], float]:
    def choose_eta(_: Callable[[Array], float], __: Array, ___: Array) -> float:
        return eta
    return choose_eta


def backtracking_learning_rate(
    beta: float = 0.8,
    max_halvings: int = 20,
) -> Callable[[Callable[[Array], float], Array, Array], float]:
    def choose_eta(f: Callable[[Array], float], x: Array, g: Array) -> float:
        eta = 1.0
        p = 1
        fx = f(x)
        gg = float(np.dot(g, g))
        while f(x - eta * g) > fx - 0.5 * eta * gg and p < max_halvings:
            eta *= beta
            p += 1
        return eta
    return choose_eta


@dataclass
class GDResult:
    x: Array
    f_value: float
    grad_norm: float
    iterations: int
    converged: bool
    diverged: bool
    last_step_norm: float


def gradient_descent(
    f: Callable[[Array], float],
    grad: Callable[[Array], Array],
    x0: Array,
    eta_strategy: Callable[[Callable[[Array], float], Array, Array], float],
    eps: float = 1e-6,
    kmax: int = 30000,
    divergence_limit: float = 1e10,
) -> GDResult:
    """
    schema de calcul din enunt
    criteriul de oprire fiind ||x_{k+1} - x_k|| = eta*||grad F(x_k)|| <= eps
    """
    x = np.asarray(x0, dtype=float).copy()
    last_step_norm = math.inf
    diverged = False

    for k in range(1, kmax + 1):
        g = grad(x)
        gnorm = norm2(g)

        if not np.all(np.isfinite(g)) or not math.isfinite(gnorm):
            diverged = True
            break

        eta = eta_strategy(f, x, g)
        step = eta * g
        last_step_norm = norm2(step)

        if not math.isfinite(last_step_norm) or last_step_norm > divergence_limit:
            diverged = True
            break

        x = x - step

        if last_step_norm <= eps:
            return GDResult(
                x=x,
                f_value=f(x),
                grad_norm=norm2(grad(x)),
                iterations=k,
                converged=True,
                diverged=False,
                last_step_norm=last_step_norm,
            )

    return GDResult(
        x=x,
        f_value=f(x) if np.all(np.isfinite(x)) else float("nan"),
        grad_norm=norm2(grad(x)) if np.all(np.isfinite(x)) else float("inf"),
        iterations=kmax,
        converged=False,
        diverged=diverged,
        last_step_norm=last_step_norm,
    )


def make_gradient(
    obj: Objective,
    mode: str,
    h: float = 1e-5,
) -> Callable[[Array], Array]:
    if mode == "analitic":
        return obj.grad
    if mode == "aproximat":
        return lambda x: finite_difference_gradient(obj.f, x, h=h)
    raise ValueError("mode trebuie sa fie 'analitic' sau 'aproximat'")


# -----------------------------
# rulare teste si afisare rezultate
# -----------------------------

def run_single_test(
    obj: Objective,
    grad_mode: str,
    eta_name: str,
    eta_strategy: Callable[[Callable[[Array], float], Array, Array], float],
    eps: float,
    h: float,
) -> Dict[str, object]:
    grad = make_gradient(obj, grad_mode, h=h)
    result = gradient_descent(obj.f, grad, obj.x0, eta_strategy, eps=eps)

    error = None
    if obj.known_minimum is not None:
        error = norm2(result.x - obj.known_minimum)

    return {
        "functie": obj.name,
        "grad": grad_mode,
        "eta": eta_name,
        "x0": obj.x0,
        "x": result.x,
        "F(x)": result.f_value,
        "||grad||": result.grad_norm,
        "iteratii": result.iterations,
        "convergent": result.converged,
        "divergent": result.diverged,
        "eroare_fata_de_x*": error,
    }


def print_result(row: Dict[str, object]) -> None:
    x = np.asarray(row["x"], dtype=float)
    err = row["eroare_fata_de_x*"]
    err_text = "-" if err is None else f"{err:.3e}"
    status = "OK" if row["convergent"] else ("DIVERGENT" if row["divergent"] else "NECONVERGENT")

    print(f"  gradient={row['grad']:<9} | eta={row['eta']:<18} | "
          f"status={status:<11} | iter={row['iteratii']:>6} | "
          f"x=({x[0]: .8f}, {x[1]: .8f}) | "
          f"F={row['F(x)']: .8e} | ||g||={row['||grad||']: .3e} | err={err_text}")


def main() -> None:
    eps = 1e-6
    h = 1e-5
    strategies: List[Tuple[str, Callable[[Callable[[Array], float], Array, Array], float]]] = [
        ("constant 1e-3", constant_learning_rate(1e-3)),
        ("constant 1e-2", constant_learning_rate(1e-2)),
        ("backtracking", backtracking_learning_rate(beta=0.8, max_halvings=20)),
    ]

    print("Tema 8 - Gradient descendent")
    print(f"eps={eps:g}, h={h:g}, kmax=30000")
    print("-" * 120)

    all_rows: List[Dict[str, object]] = []
    for obj in OBJECTIVES:
        print(f"\n{obj.name}")
        print(f"  x0=({obj.x0[0]}, {obj.x0[1]})")
        if obj.known_minimum is not None:
            print(f"  x* cunoscut/aprox. = ({obj.known_minimum[0]}, {obj.known_minimum[1]})")
        if obj.note:
            print(f"  Observatie: {obj.note}")

        for grad_mode in ("analitic", "aproximat"):
            for eta_name, eta_strategy in strategies:
                row = run_single_test(obj, grad_mode, eta_name, eta_strategy, eps=eps, h=h)
                all_rows.append(row)
                print_result(row)

    print("\n" + "-" * 120)
    print("Comparatie numar iteratii: pentru aceeasi functie si aceeasi strategie eta, comparati liniile")
    print("'gradient=analitic' cu 'gradient=aproximat'. In general, pentru h=1e-5 rezultatele sunt aproape identice")
    print("la functiile polinomiale; diferente mai vizibile pot aparea la functii neconvexe sau slab conditionate.")


if __name__ == "__main__":
    main()
