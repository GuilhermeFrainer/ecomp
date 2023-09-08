import numpy as np
import euler_method as euler
import shooting_method as sht


# Parâmetros do problema
INITIAL_Y = 0.5
FINAL_CONDITION = np.log(2)

# Parâmetros do método
Z_GUESS = -20
X_START = 1
X_STOP = 2
X_NUM = 10000 # Aumentar para uma comparação mais precisa
STEP = (X_STOP - X_START) / X_NUM


def main():
    x_space = np.linspace(X_START, X_STOP, X_NUM)
    # [x, y, z]
    dz = lambda args: 2 * np.log(args[0]) / np.power(args[0], 2) - 2 * args[1] / np.power(args[0], 2) - 4 * args[2] / args[0]
    dy = lambda args: args[2]
    solution = sht.solve_by_shooting(Z_GUESS, x_space, STEP, INITIAL_Y, FINAL_CONDITION, dz, dy)
    print(f"Chute que resolve problema: {solution}")
    print(f"T final por esse chute: {sht.attempt_shooting(solution, x_space, STEP, INITIAL_Y, dz, dy)}")
    print(f"ln(2): {np.log(2)}")


# Retorna a solução exata apresentada pelo exercício
def exact_solution(x: float) -> float:
    return (4 / x) - (2 / np.power(x, 2)) + np.log(x) - 1.5


if __name__ == "__main__":
    main()

