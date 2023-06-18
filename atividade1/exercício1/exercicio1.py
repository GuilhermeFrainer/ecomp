import matplotlib
import numpy as np
import math
import gmpy2 # Usado pois math.factorial dá overflow ao converter int para float
from tabulate import tabulate

X = math.pi / 4
N_TEST_CASES = [5, 10, 100]
TARGET_VALUE = math.sqrt(2)

def main():
    results = [2 * taylor_series_sen_x(X, test_case) for test_case in N_TEST_CASES]
    errors = [get_error(result, TARGET_VALUE) for result in results]
    
    # Resposta (a)
    print("Resposta (a) com 20 casas decimais")
    print(tabulate({
        "n": N_TEST_CASES,
        "Aproximação": results,
        "Erro (%)": errors
        },
        headers="keys",
        floatfmt=".20f"
        ))
    # TODO: Calcular erro percentual relativo


# Calcula o polinômio de Taylor que aproxima sen(x) de grau n no ponto x
def taylor_series_sen_x(x: float, n: int) -> float:
    return sum([math.pow(-1, k) * math.pow(x, 2 * k + 1) / gmpy2.fac(2 * k + 1) for k in range(n + 1)])


# Calcula o erro relativo percentual
def get_error(estimate: float, real_value: float) -> float:
    return (real_value - estimate) / real_value * 100


if __name__ == "__main__":
    main()

