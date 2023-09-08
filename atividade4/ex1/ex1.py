import numpy as np
import shooting_method as sht
import matplotlib.pyplot as plt


# Parâmetros do problema
INITIAL_Y = 0.5
FINAL_CONDITION = np.log(2)
X_START = 1
X_STOP = 2

# Parâmetros do método
Z_GUESS = -20
STEP = [1e-2, 1e-3, 1e-4]


def main():
    # args: [x, y, z]
    dz = lambda args: 2 * np.log(args[0]) / np.power(args[0], 2) - 2 * args[1] / np.power(args[0], 2) - 4 * args[2] / args[0]
    dy = lambda args: args[2]
    
    solutions = []
    for step in STEP:
        x_num = int((X_STOP - X_START) / step)
        x_space = np.linspace(X_START, X_STOP, x_num)
        solutions.append(sht.solve_by_shooting(Z_GUESS, x_space, step, INITIAL_Y, FINAL_CONDITION, dz, dy))

    for (i, sol) in enumerate(solutions):
        print(f"Solução {i + 1}: {sol}")

    # (a)
    # Calcula valor pelo método de Euler no intervalo [0, 2] para cada valor encontrado para y''
    step = 1e-3
    x_space = np.linspace(1, 2, 1000)
    sol_vecs = [] # Guarda valores de y calculados no intervalo para cada solução encontrada
    for (i, sol) in enumerate(solutions):
        vars = [[1, INITIAL_Y, sol]] # Vetor em que cada entrada é uma lista no formato [x, y, z]
        for (i, x) in enumerate(x_space[1:]):
            next_y = vars[i][1] + dy(vars[i]) * step
            next_z = vars[i][2] + dz(vars[i]) * step
            vars.append([x, next_y, next_z])
        sol_vecs.append([y for (x, y, z) in vars])

    for (i, sol) in enumerate(sol_vecs):
        plt.plot(x_space, sol, label=f"$10^{i + 2}$")
    plt.legend()
    plt.show()
    plt.savefig("Exercício 1.png")
    return


# Retorna a solução exata apresentada pelo exercício
def exact_solution(x: float) -> float:
    return (4 / x) - (2 / np.power(x, 2)) + np.log(x) - 1.5


if __name__ == "__main__":
    main()

