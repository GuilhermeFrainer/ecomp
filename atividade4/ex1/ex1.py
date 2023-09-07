import numpy as np
import euler_method as euler


# Parâmetros do problema
T0 = 300
T1 = 400

# Parâmetros do método
Z_GUESS = -13.2075
X_START = 0
X_STOP = 10
X_NUM = 10000 # Aumentar para uma comparação mais precisa
STEP = (X_STOP - X_START) / X_NUM
TOLERANCE = 1e-3


def main():
    x_space = np.linspace(X_START, X_STOP, X_NUM)
    dz = lambda args: -0.05 * (200 - args[0])
    dT = lambda args: args[1]
    solutions = solve_by_shooting(x_space, STEP, T0, T1, Z_GUESS, dz, dT)
    print(f"Final T: {solutions[-1][0]}\nFinal z: {solutions[-1][1]}")
    print(np.isclose(solutions[-1][0], 400, rtol=TOLERANCE))


# Resolve PVC pelo método do tiro
def solve_by_shooting(
    x_space: np.ndarray, # Intervalo limitado pelos "valores de contorno"
    step: float,
    initial_condition: float,
    final_condition: float,
    initial_guess: float,
    func1, # Equação 1
    func2 # Equação 2
):
    # Variáveis para recalcular chute
    recalculate_factor = 5 # Fator pelo qual recalibrar chute
    overshot_last_time = True
    
    
    i = 0
    var_history = [[initial_condition, initial_guess]]
    while i < 1:
        for (i, x) in enumerate(x_space):
            next_z = euler.iterate(var_history[i], var_history[i][1], step, func1)
            next_t = euler.iterate(var_history[i], var_history[i][0], step, func2)
            var_history.append([next_t, next_z])
        i += 1

    return var_history


# Retorna a solução exata apresentada pelo exercício
def exact_solution(x: float) -> float:
    return (4 / x) - (2 / np.power(x, 2)) + np.log(x) - 1.5


if __name__ == "__main__":
    main()

