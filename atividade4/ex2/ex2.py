import shooting_method as sht
import numpy as np

# Parâmetros do problema
K = 1
W = 1
L = 10
A = 4

X_START = 0
X_END = L
SMALL_ERROR = 1e-3
INITIAL_Y = 0 + SMALL_ERROR # Evita divisão por zero
FINAL_Y = A - SMALL_ERROR

# Parâmetros do modelo
INITIAL_GUESS = 1
STEP = 1e-3


def main():
    # args: [x, y, z]
    dz = lambda args: (A - 2 * args[1]) * (W - args[2]) * (W - (K + 1) * args[2]) / (K * args[1] * (A - args[1]))
    dy = lambda args: args[2]

    x_num = int((X_END - X_START) / STEP)
    x_space = np.linspace(X_START, X_END, x_num)

    solution = sht.solve_by_shooting(
        INITIAL_GUESS,
        x_space,
        STEP,
        INITIAL_Y,
        FINAL_Y,
        dz,
        dy
    )
    if solution.success == False:
        print("Scipy não conseguiu minimizar erro.")
        print(solution)
    
    y = sht.attempt_shooting(
        solution.x[0],
        x_space,
        STEP,
        INITIAL_Y,
        dz,
        dy
    )

    if solution.success == False:
        print("")
    
    print(f"Solution vec: {solution.x}")
    print(f"y na solução: {y}")

    """
    TENTATIVA COM SCIPY
    Também deu errado

    p = [K, W, L, A]
    x = np.linspace(0, L, 10)
    y = np.zeros((2, x.size))
    y += SMALL_ERROR
    x_plot = np.linspace(X_START, L, 100)
    sol = scint.solve_bvp(func, bc, x, y, p)
    print(sol)
    """


def optimal_solution():
    ...


# Funções abaixo foram escritas para a tentativa de resolver 

# p: [k, W, L, A]
def func(x, y, p):
    return np.vstack((
        y[1],
        (p[3] - 2 * y[0]) * (p[1] - y[1]) * (p[1] - (p[0] + 1) * y[1]) / (p[0] * y[0] * (p[3] - y[0]))
    ))


def bc(ya, yb, p):
    return np.array([ya[0], yb[0], p[0], p[1], p[2], p[3]])


if __name__ == "__main__":
    main()

