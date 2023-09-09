# Implementa método do tiro para PVC de duas equações
import numpy as np
import scipy.optimize as opt
import euler_method as euler


# Resolve PVC pelo método do tiro
def solve_by_shooting(initial_guess: float, *args) -> float:
    """
    Resolve PVC pelo método do tiro para sistemas de duas equações.

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
    if np.isclose(solution, final_condition):
        return initial_guess
    # Caso contrário, usa scipy.optimize.fsolve() na função de resíduo para encontrar chute que
    # minimiza erro. Sugestão de CHAPRA (2018) para PVCs em sistemas de equações diferenciais
    # não-lineares.
    else:
        func_args = (x_space, step, initial_condition, final_condition, target_function, mock_function)
        scipy_solution = opt.minimize(residue, initial_guess, args=func_args)
        # Devolve "apenas" o primeiro item, pois solução é um array contendo apenas uma variável
        return scipy_solution


def attempt_shooting(initial_guess: float, *args) -> float:
    """
    Tentativa de resolver PVC pelo método do tiro para sistemas de duas equações.

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
    var_history = [[x_space[0], initial_condition, initial_guess]]
    for (i, x) in enumerate(x_space):
        next_z = euler.iterate(var_history[i], var_history[i][2], step, target_function)
        next_y = euler.iterate(var_history[i], var_history[i][1], step, mock_function)
        var_history.append([x, next_y, next_z])
    return var_history[-1][1]


# Função de erro a ter raízes encontradas pelo scipy.optimize.fsolve().
# Apenas chama 'attempt shooting' e subtrai pelo valor na condição de contorno.
def residue(initial_guess: float, *args) -> float:
    x_space, step, initial_condition, final_condition, target_function, mock_function = args
    attempt_result = attempt_shooting(initial_guess[0], x_space, step, initial_condition, target_function, mock_function)
    return abs(attempt_result - final_condition)

