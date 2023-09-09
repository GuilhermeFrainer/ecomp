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


# Testes
import unittest


# Testa utilizando exemplo 22.1 de CHAPRA (2018)
class TestEuler(unittest.TestCase):
    def test_iterate(self):
        initial_condition = [2, 0]
        func = lambda x: 4 * np.exp(0.8 * x[1]) - 0.5 * x[0]
        
        iter_result = iterate(initial_condition, initial_condition[0], 1, func)
        self.assertAlmostEqual(iter_result, 5)

    def test_solve(self):
        initial_condition = [2, 0]
        step = 1
        t_space = np.linspace(0, 4, 5)
        func = lambda x: 4 * np.exp(0.8 * x[1]) - 0.5 * x[0]

        solutions = solve(t_space, initial_condition, step, func)
        expected_solutions = [
            2.0,
            5.0,
            11.40216,
            25.51321,
            56.84931
        ]
        for (sol, exp) in zip(solutions, expected_solutions):
            self.assertAlmostEqual(sol[0], exp, places=5)


if __name__ == "__main__":
    unittest.main()

