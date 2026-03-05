# EXERCITIUL 1
# Determina cel mai mic u = 10^(-m) astfel incat 1.0 + u != 1.0

def machine_epsilon():
    m = 0
    while True:
        u = 10.0 ** (-m)
        if (1.0 + u) != 1.0:
            return u, m
        m += 1


if __name__ == "__main__":
    u, m = machine_epsilon()

    print("Cel mai mic u = 10^(-m) astfel incat 1.0 + u != 1.0")
    print(f"m = {m}")
    print(f"u = {u:.20e}")
    print()

    print("Verificari:")
    print("1.0 + u     =", 1.0 + u)
    print("1.0 + u/10  =", 1.0 + u/10)