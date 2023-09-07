import numpy as np
import euler_method as euler
import scipy.optimize as opt


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
    solution = solve_by_shooting(Z_GUESS, x_space, STEP, T0, T1, dz, dT)
    print(f"Final T: {solution}")
    print(f"Outra solução: T: {residue(Z_GUESS, T1, x_space, STEP, T0, dz, dT)}")


# Resolve PVC pelo método do tiro
def solve_by_shooting(initial_guess: float, *args) -> float:
    """
    Resolve PVC pelo método do tiro.

    
    Argumentos
    ----------
    x_space : np.ndarray
        Intervalo delimitado pelos "valores de contorno".
    step : float
    initial_condition : float
    final_condition : float
    func1 : callable ``f(x)``
        Equação 1
    func2 : callable ``f(x)``
        Equação 2

    Retorna
    --------
    x : float
        "Chute" que resolve PVC.
    """
    x_space, step, initial_condition, final_condition, func1, func2 = args
    
    # Tenta resolver PVC com chute inicial
    solution = attempt_shooting(initial_guess, x_space, step, initial_condition, func1, func2)

    # Retorna o chute inicial caso seja bem-sucedido
    if np.isclose(solution, final_condition, rtol=TOLERANCE):
        return initial_guess
    # Caso contrário, usa scipy.optimize.fsolve() na função de resíduo para encontrar chute que
    # minimiza erro. Sugestão de CHAPRA (2018) para PVCs em sistemas de equações diferenciais
    # não-lineares.
    else:
        return "Fracasso"


def attempt_shooting(initial_guess: float, *args) -> float:
    """
    Tentativa de resolver PVC pelo método do tiro.

    Argumentos
    ----------
    x_space : np.ndarray
        Intervalo delimitado pelos "valores de contorno".
    step : float
    initial_condition : float
    func1 : callable ``f(x)``
        Equação 1
    func2 : callable ``f(x)``
        Equação 2

    Retorna
    --------
    x : float
        Solução. Retorna valor a ser comparado com as condições de contorno.
    """
    x_space, step, initial_condition, func1, func2 = args
    var_history = [[initial_condition, initial_guess]]
    for (i, _) in enumerate(x_space):
        next_z = euler.iterate(var_history[i], var_history[i][1], step, func1)
        next_t = euler.iterate(var_history[i], var_history[i][0], step, func2)
        var_history.append([next_t, next_z])
    return var_history[-1][0]


# Função de erro a ser minimizada pelo scipy.optimize.fsolve().
# Apenas chama 'attempt shooting' e subtrai pelo valor na condição de contorno.
def residue(initial_guess: float, final_condition: float, *args) -> float:
    x_space, step, initial_condition, func1, func2 = args
    attempt_result = attempt_shooting(initial_guess, x_space, step, initial_condition, func1, func2)
    return attempt_result - final_condition


# Retorna a solução exata apresentada pelo exercício
def exact_solution(x: float) -> float:
    return (4 / x) - (2 / np.power(x, 2)) + np.log(x) - 1.5


if __name__ == "__main__":
    main()

