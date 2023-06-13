import matplotlib
import numpy
import math

X = math.pi / 4

def main():
    print(f"n = 5: {taylor_series(X, 5)}\nn = 10: {taylor_series(X, 10)}\nn = 100: {taylor_series(X, 100)}")



# Calcula o polinômio de Taylor de grau n no ponto x
def taylor_series(x: float, n: int) -> float:
    return sum([math.pow(-1, k) * math.pow(x, 2 * k + 1) / math.factorial(2 * k + 1) for k in range(n + 1)])


if __name__ == "__main__":
    main()

