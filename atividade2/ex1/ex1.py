from typing import Callable
import math
import matplotlib.pyplot as plt
import numpy as np


ALPHA = 0.1
MAX_N = 0.8
MAX_ITERATIONS = 100
STARTING_N = 0.1
MAX_ERROR = 10 ** -5


def main():
    B = np.linspace(0.1, 5)
    # (i)
    f1 = lambda x: min(ALPHA * x, MAX_N)

    # (ii)
    f2 = lambda x: min(ALPHA * math.sqrt(x), MAX_N)

    # (iii)
    f3 = lambda x: min(ALPHA * math.pow(x, 2), MAX_N)

    f1_series = [pred(b, f1) for b in B]
    f2_series = [pred(b, f2) for b in B]
    f3_series = [pred(b, f3) for b in B]
    
    plt.plot(B, f1_series, 'ob')
    #plt.plot(B, f2_series)
    #plt.plot(B, f3_series)
    plt.show()


# Retorna T_inf para um determinado B e uma função para n_t
def pred(B: float, func: Callable) -> float:
    curr_error = MAX_ERROR + 1 # Garante que o loop rode
    T = [STARTING_N]
    i = 0
    # Usa erro para verificar se equilíbrio de longo prazo foi atingido
    while curr_error > MAX_ERROR:
        T.append(B / (1 - func(T[i])))
        curr_error = abs((T[i + 1] - T[i]) / T[i])
        i += 1
    return T[i]


if __name__=="__main__":
    main()

