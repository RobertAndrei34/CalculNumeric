def machine_epsilon():
    m = 0
    while True:
        u = 10.0 ** (-m)
        if (1.0 + u) != 1.0:
            return u
        m += 1


def neasociativitate_adunare():
    u = machine_epsilon()

    x = 1.0
    y = u / 10.0
    z = u / 10.0

    stanga = (x + y) + z
    dreapta = x + (y + z)

    print("=== Neasociativitate Adunare ===")
    print("x =", x)
    print("y =", y)
    print("z =", z)
    print("(x + y) + z =", stanga)
    print("x + (y + z) =", dreapta)
    print("Sunt diferite?", stanga != dreapta)
    print()


def neasociativitate_inmultire():
    x = 1e308
    y = 1e308
    z = 1e-308

    stanga = (x * y) * z
    dreapta = x * (y * z)

    print("=== Neasociativitate Inmultire ===")
    print("x =", x)
    print("y =", y)
    print("z =", z)
    print("(x * y) * z =", stanga)
    print("x * (y * z) =", dreapta)
    print("Sunt diferite?", stanga != dreapta)


if __name__ == "__main__":
    neasociativitate_adunare()
    neasociativitate_inmultire()