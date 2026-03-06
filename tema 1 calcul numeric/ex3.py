import math
import random
import time


# reducere in (-pi/2, pi/2]

def reduce_interval(x):
    return ((x + math.pi/2) % math.pi) - math.pi/2


# Fractii continue

def tan_continua(x, eps=1e-12, max_iter=10000):
    x = reduce_interval(x)

    if abs(abs(x) - math.pi/2) < 1e-15:
        return math.nan

    x2 = x * x
    a = -x2

    b0 = 1.0
    f = b0
    C = f
    D = 0.0

    for j in range(1, max_iter):
        bj = 2*j + 1

        D = bj + a * D
        if D == 0:
            D = 1e-12
        D = 1.0 / D

        C = bj + a / C
        if C == 0:
            C = 1e-12

        delta = C * D
        f *= delta

        if abs(delta - 1) < eps:
            break

    return x / f


# Polinom Maclaurin

def tan_polinom(x):
    x = reduce_interval(x)

    if abs(abs(x) - math.pi/2) < 1e-15:
        return math.nan

    if abs(x) > math.pi/4:
        x_reduced = math.pi/2 - abs(x)
        if x > 0:
            result = tan_polinom_core(x_reduced)
            return 1.0 / result if abs(result) > 1e-15 else math.nan
        else:
            result = tan_polinom_core(x_reduced)
            return -1.0 / result if abs(result) > 1e-15 else math.nan
    else:
        return tan_polinom_core(x)


def tan_polinom_core(x):
    c1 = 0.3333333333333333
    c2 = 0.13333333333333333
    c3 = 0.053968253968254
    c4 = 0.0218694885361552

    x_2 = x * x
    x_3 = x_2 * x
    x_4 = x_2 * x_2
    x_6 = x_4 * x_2

    return x + x_3*(c1+(c2*x_2)+(c3*x_4)+(c4*x_6))

# evaluare statistica

def is_zero_like(diff, truth, atol=1e-12, rtol=1e-10):
    return diff <= (atol + rtol * abs(truth))


def evaluare(N=10000):
    valori = [random.uniform(-math.pi/2 + 1e-8, math.pi/2 - 1e-8) for _ in range(N)]

    eroare_cf = 0.0
    eroare_poly = 0.0
    zero_cf = 0
    zero_poly = 0
    max_error_poly = 0.0
    max_error_cf = 0.0

    t_start_cf = time.time()
    for x in valori:
        real = math.tan(x)
        aprox_cf = tan_continua(x)
        diff_cf = abs(real - aprox_cf)
        eroare_cf += diff_cf
        max_error_cf = max(max_error_cf, diff_cf)

        if is_zero_like(diff_cf, real, atol=1e-12, rtol=1e-10):
            zero_cf += 1
    t_end_cf = time.time()
    timp_cf = t_end_cf - t_start_cf

    # timing for polynomial
    t_start_poly = time.time()
    for x in valori:
        real = math.tan(x)
        aprox_poly = tan_polinom(x)
        diff_poly = abs(real - aprox_poly)
        eroare_poly += diff_poly
        max_error_poly = max(max_error_poly, diff_poly)

        if is_zero_like(diff_poly, real, atol=1e-6, rtol=1e-4):
            zero_poly += 1
    t_end_poly = time.time()
    timp_poly = t_end_poly - t_start_poly

    print("FRACTII CONTINUE:")
    print("  Diferente ~0 (±0)     =", zero_cf, "din", N)
    print("  Timp calcul (10000)   =", f"{timp_cf:.6f} secunde")
    print()
    print("POLINOM MACLAURIN:")
    print("  Diferente ~0 (±0)     =", zero_poly, "din", N)
    print("  Timp calcul (10000)   =", f"{timp_poly:.6f} secunde")
    print()

if __name__ == "__main__":
    evaluare()
