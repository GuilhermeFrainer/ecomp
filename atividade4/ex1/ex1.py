import scipy.integrate as scint
import numpy as np
import euler_method as euler


STEP = 1
INITIAL_CONDITION = [2, 0]

def main():
    t_space = np.linspace(0, 4, 5)
    func = lambda x: 4 * np.exp(0.8 * x[1]) - 0.5 * x[0]
    
    
    solutions = euler.solve(t_space, INITIAL_CONDITION, STEP, func)

    for (y, t) in solutions:
        print(f"t: {t}\ty: {y}")


# Retorna a solução exata apresentada pelo exercício
def exact_solution(x: float) -> float:
    return (4 / x) - (2 / np.power(x, 2)) + np.log(x) - 1.5


if __name__ == "__main__":
    main()

