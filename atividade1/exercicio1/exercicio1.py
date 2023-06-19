# Arquivos externos
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import math
import gmpy2 # Usado pois math.factorial dá overflow ao converter int para float
from tabulate import tabulate

# Arquivos internos
import pi_formatter


X = math.pi / 4
N_TEST_CASES = [5, 10, 100]
TARGET_VALUE = math.sqrt(2)


def main():
    results = [2 * taylor_series_sen_x(X, test_case) for test_case in N_TEST_CASES]
    errors = [get_relative_error(result, TARGET_VALUE) for result in results]
    
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
    
    # Resposta (b)
    # Vetoriza a função, para que possa receber um array como argumento
    taylor_vec = np.vectorize(taylor_series_sen_x)
    
    x_axis = np.arange(-np.pi, np.pi, 0.01)
    
    figure = plt.figure("Exercício 1b")
    axis = figure.add_subplot(111)

    axis.plot(x_axis, np.sin(x_axis), '#000000', label="$sen(x)$")
    axis.plot(x_axis, taylor_vec(x_axis, N_TEST_CASES[0]), '#ff0000', label="$P_{5}(x)$")
    axis.plot(x_axis, taylor_vec(x_axis, N_TEST_CASES[1]), '#00ff00', label="$P_{10}(x)$")
    axis.plot(x_axis, taylor_vec(x_axis, N_TEST_CASES[2]), '#0000ff', label="$P_{100}(x)$")

    # Dá título aos eixos e ao gráfico
    axis.set_title("Polinômio de Taylor que aproxima sen(x)")
    axis.set_xlabel("x")
    axis.set_ylabel("f (x)")

    axis.xaxis.set_major_locator(tck.MultipleLocator(np.pi / 2))
    axis.xaxis.set_minor_locator(tck.MultipleLocator(np.pi / 4))
    axis.xaxis.set_major_formatter(plt.FuncFormatter(pi_formatter.multiple_formatter()))

    # Coloca legenda
    axis.legend()

    figure.savefig("Exercício 1b.png")

    # Resposta (c)
    figure = plt.figure("Exercício 1c")
    axis = figure.add_subplot(111)

    axis.plot(
        x_axis,
        np.absolute(np.sin(x_axis) - taylor_vec(x_axis, N_TEST_CASES[0])),
        "#ff0000",
        label="$E_{5}$"
    )
    axis.plot(
        x_axis,
        np.absolute(np.sin(x_axis) - taylor_vec(x_axis, N_TEST_CASES[1])),
        "#00ff00",
        label="$E_{10}$"
    )
    axis.plot(
        x_axis,
        np.absolute(
            np.sin(x_axis) - taylor_vec(x_axis, N_TEST_CASES[2]),
        ),
        "#0000ff",
        label="$E_{100}$"
    )

    # Dá título aos eixos e ao gráfico
    axis.set_title("Erros nas estimativas de sen(x)")
    axis.set_xlabel("x")
    axis.set_ylabel("E (x)")

    axis.xaxis.set_major_locator(tck.MultipleLocator(np.pi / 2))
    axis.xaxis.set_minor_locator(tck.MultipleLocator(np.pi / 4))
    axis.xaxis.set_major_formatter(plt.FuncFormatter(pi_formatter.multiple_formatter()))

    # Coloca eixo y em escala logarítmica
    axis.set_yscale("log")
    
    # Coloca legenda
    axis.legend()

    #plt.show()

    figure.savefig("Exercício 1c.png")


# Calcula o polinômio de Taylor que aproxima sen(x) de grau n no ponto x
def taylor_series_sen_x(x: float, n: int) -> float:
    return sum([np.power(-1, k) * np.power(x, 2 * k + 1) / gmpy2.fac(2 * k + 1) for k in range(n + 1)])


# Calcula o erro relativo percentual
def get_relative_error(estimate: float, real_value: float) -> float:
    return (real_value - estimate) / real_value * 100


if __name__ == "__main__":
    main()

