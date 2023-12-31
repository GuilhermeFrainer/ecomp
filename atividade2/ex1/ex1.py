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
    B_2 = np.linspace(0.1, 20)
    # (i)
    f1 = lambda x: min(ALPHA * x, MAX_N)

    # (ii)
    f2 = lambda x: min(ALPHA * math.sqrt(x), MAX_N)
    # Usa-se B maior para o dois pois o ponto de ruptura acontece mais tarde
    B_2 = np.linspace(0.1, 20)

    # (iii)
    f3 = lambda x: min(ALPHA * math.pow(x, 2), MAX_N)

    # Salva equilíbrios de longo prazo em função de B
    f_series = [
        [pred(b, f1) for b in B],
        [pred(b, f2) for b in B_2],
        [pred(b, f3) for b in B]
    ]
    
    # Apresenta gráficos do equilíbrio de longo prazo em função de B
    fig, axs = plt.subplots(3)
    axs[0].plot(B, f_series[0], 'ob')
    axs[0].set_title("(i)")
    axs[1].plot(B_2, f_series[1], 'ob')
    axs[1].set_title("(ii)")
    axs[2].plot(B, f_series[2], 'ob')
    axs[2].set_title("(iii)")
    for ax in axs:
        ax.set_xlabel("$B$")
        ax.set_ylabel("$T_\infty$")

    plt.tight_layout()
    plt.show()
    fig.savefig("Equilíbrios de longo prazo.png")

    # (iv)
    T = np.linspace(0, 2)
    nt_series = [
        [f1(t) for t in T],
        [f2(t) for t in T],
        [f3(t) for t in T]
    ]
    fig, axs = plt.subplots()
    labels = [
        '$min\{\\alpha T_t, \\bar{n}\}$',
        '$min\{\\alpha\\sqrt{T_t}, \\bar{n}\}$',
        '$min\{\\alpha T_t^2, \\bar{n}\}$'
    ]
    for (i, series) in enumerate(nt_series):
        axs.plot(T, series, label=labels[i])

    axs.legend()
    plt.show()
    fig.savefig("Funções nt.png")


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
    return T[-1]


if __name__=="__main__":
    main()

