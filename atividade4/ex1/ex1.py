import numpy as np
import euler_method as euler
import scipy.optimize as opt


# Parâmetros do problema
INITIAL_Y = 300
FINAL_CONDITION = 400

# Parâmetros do método
Z_GUESS = -20.2075
X_START = 0
X_STOP = 10
X_NUM = 10000 # Aumentar para uma comparação mais precisa
STEP = (X_STOP - X_START) / X_NUM
TOLERANCE = 1e-3


def main():
    x_space = np.linspace(X_START, X_STOP, X_NUM)
    # [x, y, z]
    dz = lambda args: -0.05 * (200 - args[1])
    dy = lambda args: args[2]
    solution = solve_by_shooting(Z_GUESS, x_space, STEP, INITIAL_Y, FINAL_CONDITION, dz, dy)
    print(f"Chute que resolve problema: {solution}")
    print(f"T final por esse chute: {attempt_shooting(solution, x_space, STEP, INITIAL_Y, dz, dy)}")


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
    target_function : callable ``f(x)``
        Equação do problema.
    mock_function : callable ``f(x)``
        Equação criada para resolver o problema.

    Retorna
    --------
    x : float
        "Chute" que resolve PVC.
    """
    x_space, step, initial_condition, final_condition, target_function, mock_function = args
    
    # Tenta resolver PVC com chute inicial
    solution = attempt_shooting(initial_guess, x_space, step, initial_condition, target_function, mock_function)

    # Retorna o chute inicial caso seja bem-sucedido
    if np.isclose(solution, final_condition, rtol=TOLERANCE):
        return initial_guess
    # Caso contrário, usa scipy.optimize.fsolve() na função de resíduo para encontrar chute que
    # minimiza erro. Sugestão de CHAPRA (2018) para PVCs em sistemas de equações diferenciais
    # não-lineares.
    else:
        func_args = (x_space, step, initial_condition, final_condition, target_function, mock_function)
        scipy_solution = opt.fsolve(residue, initial_guess, args=func_args)
        return scipy_solution


def attempt_shooting(initial_guess: float, *args) -> float:
    """
    Tentativa de resolver PVC pelo método do tiro.

    Argumentos
    ----------
    x_space : np.ndarray
        Intervalo delimitado pelos "valores de contorno".
    step : float
    initial_condition : float
    target_function : callable ``f(x)``
        Equação do problema.
    mock_function : callable ``f(x)``
        Equação criada para resolver o problema.

    Retorna
    --------
    x : float
        Solução. Retorna valor a ser comparado com as condições de contorno.
    """
    x_space, step, initial_condition, target_function, mock_function = args
    var_history = [[X_START, initial_condition, initial_guess]]
    for (i, x) in enumerate(x_space):
        next_z = euler.iterate(var_history[i], var_history[i][2], step, target_function)
        next_y = euler.iterate(var_history[i], var_history[i][1], step, mock_function)
        var_history.append([x, next_y, next_z])
    return var_history[-1][1]


# Função de erro a ser minimizada pelo scipy.optimize.fsolve().
# Apenas chama 'attempt shooting' e subtrai pelo valor na condição de contorno.
def residue(initial_guess: float, *args) -> float:
    x_space, step, initial_condition, final_condition, target_function, mock_function = args
    attempt_result = attempt_shooting(initial_guess, x_space, step, initial_condition, target_function, mock_function)
    return attempt_result - final_condition


# Retorna a solução exata apresentada pelo exercício
def exact_solution(x: float) -> float:
    return (4 / x) - (2 / np.power(x, 2)) + np.log(x) - 1.5


if __name__ == "__main__":
    main()

