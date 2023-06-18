import matplotlib
import numpy
import math
import gmpy2 # Usado pois math.factorial dá overflow ao converter int para float
from tabulate import tabulate

X = math.pi / 4
N_TEST_CASES = [5, 10, 100]

def main():
    results = [2 * taylor_series_sen_x(X, test_case) for test_case in N_TEST_CASES]
    print(tabulate({
        "n": N_TEST_CASES,
        "Aproximação": results
        },
        headers="keys",
        floatfmt=".20f"
        ))
    # TODO: Calcular erro percentual relativo




# Calcula o polinômio de Taylor que aproxima sen(x) de grau n no ponto x
def taylor_series_sen_x(x: float, n: int) -> float:
    return sum([math.pow(-1, k) * math.pow(x, 2 * k + 1) / gmpy2.fac(2 * k + 1) for k in range(n + 1)])


if __name__ == "__main__":
    main()

