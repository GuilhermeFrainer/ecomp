import shooting_method as sht
import numpy as np
import scipy.integrate as scint
import matplotlib.pyplot as plt


# Parâmetros do problema
K = [1, 1, 2, 10]
W = [1, 1, 1, 1]
L = [10, 10, 10, 10]
A = [4, 6, 6, 6]

X_START = 0
SMALL_ERROR = 1e-3
INITIAL_Y = 0 + SMALL_ERROR # Evita divisão por zero

# Parâmetros do método
INITIAL_GUESS = 1
STEP = 1e-3


def main():
    solutions = []
    # Roda método do tiro para cada cenário
    for (j, (k, w, l, a)) in enumerate(zip(K, W, L, A)):
        # Constantes que variam a cada cenário
        final_y = a - SMALL_ERROR
        x_end = l
        
        # args: [x, y, z]
        dz = lambda args: (a - 2 * args[1]) * (w - args[2]) * (w - (k + 1) * args[2]) / (k * args[1] * (a - args[1]))
        dy = lambda args: args[2]

        x_num = int((x_end - X_START) / STEP)
        x_space = np.linspace(X_START, x_end, x_num)

        solution = sht.solve_by_shooting(
            INITIAL_GUESS,
            x_space,
            STEP,
            INITIAL_Y,
            final_y,
            dz,
            dy
        )
        
        if solution.success == False:
            print("Scipy não conseguiu minimizar erro.")
        
        print("Solução pelo método do tiro:\n")
        print(solution)
        
        """
        # TENTATIVA COM SCIPY
        # Também deu errado
        def func(x, y):
            return np.vstack((
                y[1],
                (a - 2 * y[0]) * (w - y[1]) * (w - (k + 1) * y[1]) / (k * y[0] * (a - y[0]))
            ))


        def bc(ya, yb):
            return np.array([ya[0], yb[0] - a])

        x = np.linspace(0, l, 10)
        y = np.zeros((2, x.size))
        y += SMALL_ERROR
        solution = scint.solve_bvp(func, bc, x, y)
        print("Solução pela função 'solve_bvp' do scipy:\n")
        print(solution)

        # Final da tentativa pelo scipy
        """

        sol = solution.x[0]
    
        x_num = int((x_end - X_START) / STEP)
        x_space = np.linspace(X_START, x_end, x_num)
        vars = [[X_START, INITIAL_Y, sol]] # Vetor em que cada entrada é uma lista no formato [x, y, z]
        for (i, x) in enumerate(x_space[1:]):
            next_y = vars[i][1] + dy(vars[i]) * STEP
            next_z = vars[i][2] + dz(vars[i]) * STEP
            vars.append([x, next_y, next_z])
        sol_vec = [y for (x, y, z) in vars]

        plt.plot(x_space, sol_vec)
        
        plt.title(f"$w*$ para o caso ({chr(j + 97)})")
        plt.savefig(f"Exercício 2{chr(j + 97)}.png")
        plt.show()
    return


if __name__ == "__main__":
    main()

