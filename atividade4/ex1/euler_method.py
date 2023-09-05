# Implementa método de Euler para resolver PVI

import numpy as np


# Retorna passo i + 1 pelo método de euler
# Func deve ter forma func(args)
# Sendo 'args' um array-like
def iterate(args: np.ndarray, y: float, step: float, func) -> float:
    return y + func(args) * step


# Resolve PVI pelo método de Euler
# Retorna lista com solução em cada ponto de 't_space'
# Na forma [y(t), t]
def solve(t_space: np.ndarray, initial_condition: list[float], step: float, func):
    x = [initial_condition]
    
    for (i, t) in enumerate(t_space[1:]):
        next_item = iterate(x[i], x[i][0], step, func)
        x.append([next_item, t])
    return x

