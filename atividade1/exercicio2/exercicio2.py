import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
from tabulate import tabulate
import math


INITIAL_CONDITION = 1
MAX_ITERATIONS = 100
MAX_ERROR = 10 ** -5


def main():
    x = np.linspace(-3, 3, 1000)

    # Cria o polinômio pelo numpy
    p = poly.Polynomial([-3, 0, -3, 0, 1])

    # Resposta (a)
    plot_function_chart(x, p)

    # Resposta (b)
    fixed_point_roots = roots_by_fixed_point()
    plot_convergence_chart(fixed_point_roots)

    # Faz o mesmo gráfico para a raiz negativa
    #negative_roots = np.array([-x for x in fixed_point_roots])
    #plot_convergence_chart(negative_roots)

    # Resposta (c)
    # Calcula as raízes do polinômio com ferramentas no numpy
    roots = p.roots()
    # Filtra raízes complexas
    real_roots = roots[~np.iscomplex(roots)]
    real_roots = np.real(real_roots)
    print(tabulate({
        "Raízes": real_roots,
        },
        headers="keys",
        floatfmt=".5f"
    ))


# Função usada pelo exercício
def target_function(x: np.ndarray):
    return np.power(x, 4) - 3 * np.power(x, 2) - 3


# Faz gráfico da função
def plot_function_chart(x: np.ndarray, func):
    figure = plt.figure("Exercício 2a")
    axis = figure.add_subplot(111)

    # Adiciona linha horizontal em y = 0
    plt.axhline(0, color="#000000", linestyle='--')

    # Dá títulos aos eixos
    axis.set_xlabel("x")
    axis.set_ylabel("f (x)")

    axis.plot(x, func(x))

    # Indica raízes no gráfico
    axis.annotate(
        "Raiz",
        xy=(-1.9, 1),
        xytext=(-1, 10),
        arrowprops={
            "facecolor": "black",
        }
    )
    axis.annotate(
        "Raiz",
        xy=(1.9, 1),
        xytext=(1, 10),
        arrowprops={
            "facecolor": "black",
        }
    )

    figure.savefig("Exercício 2a.png")


# Calcula as raízes da função por ponto fixo
# Usa g(x) = ± (3x^2 + 3)^1/4
# Uma raiz vai ser positiva, a outra, negativa
def roots_by_fixed_point() -> np.ndarray:
    func = lambda x: math.pow(3 * math.pow(x, 2) + 3, 1/4)

    # Preparação para o algoritmo
    x  = np.array([INITIAL_CONDITION], dtype=np.float64) # Array que será retornado
    i = 1

    while i < MAX_ITERATIONS:
        new_value = func(x[i - 1])
        print(new_value)
        x = np.append(x, new_value)
        
        if abs(x[i] - x[i - 1]) < MAX_ERROR:
            print(f"Iterações até convergir: {i}")
            return x
            
        i += 1


# Desenha gráfico de convergência do método do ponto fixo
def plot_convergence_chart(fixed_point_roots: np.ndarray):
    figure = plt.figure("Exercício 2b")
    axis = figure.add_subplot(211)

    axis.set_xlabel("x")
    axis.set_ylabel("g (x)")

    axis.plot(fixed_point_roots[:-1], fixed_point_roots[1:], 's')

    # Faz o mesmo gráfico para a raiz negativa
    neg_axis = figure.add_subplot(212)

    axis.set_label("x")
    axis.set_ylabel("g (x)")

    # Converte as estimativas de raízes para seu oposto
    # Isso pode ser feito devido ao polinômio escolhido
    negative_roots = np.array([-x for x in fixed_point_roots])

    neg_axis.plot(negative_roots[:-1], negative_roots[1:], 's', color="#ff0000")

    figure.savefig("Exercício 2b.png")


if __name__ == "__main__":
    main()

